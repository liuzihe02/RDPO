import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..dpo.trainer import CustomDPOTrainer
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomRDPOTrainer(CustomDPOTrainer):
    @override
    def __init__(self, *args, reasoning_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_weight = reasoning_weight

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[
        "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"
    ]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.


        this splits up a batch tensor into the relevant stuff

        note that this includes a reasoning column for the third column

        Otherwise the average log probabilities.

        the batch here already contains logits for chosen, rejected, reasoning
        the way this is structured is defined in RDPOPairwiseDataCollatorWithPadding
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # this batch will be a dictionary of stuff decided in data_collator.py
        # dict keys are input_ids, attention_mask, labels
        # each dict item is a tensor of size (batch_size_per_device *3, seq_len)
        # since (chosen, rejected, reasoning)
        # print(f"zihe check batch shape is {batch['labels'].shape}")

        # this may cause cuda memory issues, reduce batch size to alleviate this problem
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        # this is for labels only
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # split along the first dimension
        # first third is chosen, next third is rejected, last third is reasoning
        batch_size = batch["input_ids"].size(0) // 3
        # i believe the following is the logits/logps for labels only
        chosen_logps, rejected_logps, reasoning_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits, reasoning_logits = all_logits.split(batch_size, dim=0)
        # these lengths are all the same
        chosen_length, _, _ = valid_length.split(batch_size, dim=0)

        # default loss is sigmoid here
        # last one is a normalized chosen_logps
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            # order of first 2, the chosen and rejected logps must stay the same
            return (
                chosen_logps,
                rejected_logps,
                reasoning_logps,
                chosen_logits,
                rejected_logits,
                reasoning_logits,
                chosen_logps,
            )
        else:
            return (
                chosen_logps,
                rejected_logps,
                reasoning_logps,
                chosen_logits,
                rejected_logits,
                reasoning_logits,
                chosen_logps / chosen_length,
            )

    # the compute reference log probs can remain the same
    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the RDPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_reasoning_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_reasoning_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        # logps of shape (batch,)
        print(f"zihe check shape of policy chosen logps is{policy_chosen_logps.shape}")
        print(f"zihe check shape of policy rejected logps is{policy_rejected_logps.shape}")
        print(f"zihe check shape of policy reasoning logps is{policy_reasoning_logps.shape}")
        # logits of shape (batch, sequence, d_model)
        print(f"zihe check shape of policy chosen logits is{policy_chosen_logits.shape}")
        print(f"zihe check shape of policy rejected logits is{policy_rejected_logits.shape}")
        print(f"zihe check shape of policy reasoning logits is{policy_reasoning_logits.shape}")

        # Get reference log probs
        # since compute_reference_log_probs calls a method that we override,
        # which is concatenated_forward
        # we must be careful to keep the order of arguments to allow this to work
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)

        print(f"zihe check shape of reference chosen logps is{reference_chosen_logps.shape}")

        # Compute dpo preference loss
        # this takes in logprobs, which is already averaged over sequence
        dpo_losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        # check dpo loss shape, which is simply (batch,)
        print(f"zihe check dpo loss shape is {dpo_losses.shape}")

        # not sure why this is included in default DPO implementation for LLaMA-Factory but some form of sft is included here
        # default ftx_gamma is zero
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            dpo_losses += self.ftx_gamma * sft_loss

        # For reasoning loss, we simply use the negative log probability
        # This is simpler than the DPO loss formulation because we're not comparing two alternatives
        # We just want to maximize the probability of generating good verification reasoning
        reasoning_loss = -policy_reasoning_logps

        # Combine DPO loss with reasoning loss
        combined_losses = (1 - self.reasoning_weight) * dpo_losses + self.reasoning_weight * reasoning_loss

        # IMPORTANT take mean over everything
        combined_losses = combined_losses.mean()

        # Add metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()

        # Add reasoning loss to metrics
        metrics[f"{prefix}reasoning_loss"] = reasoning_loss.mean().detach().item()
        metrics[f"{prefix}dpo_loss"] = dpo_losses.mean().item()
        metrics[f"{prefix}total_loss"] = combined_losses.item()

        # Add special metrics for ORPO
        if self.loss_type == "orpo":
            sft_loss = -policy_chosen_logps_avg
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((dpo_losses - sft_loss) / self.beta).mean().item()

        return combined_losses, metrics
