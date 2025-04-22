import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
import os

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..dpo.trainer import CustomDPOTrainer
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


class CustomRDPOTrainer(CustomDPOTrainer):
    @override
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        super().__init__(model, ref_model, finetuning_args, processor, disable_dropout, **kwargs)
        self.reasoning_weight = finetuning_args.reasoning_weight

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
    ]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.


        this splits up a batch tensor into the relevant stuff

        note that this includes a reasoning column for the third column

        Otherwise the average log probabilities.

        the batch here already contains logits for chosen, rejected, reasoning
        the way this is structured is defined in RDPOPairwiseDataCollatorWithPadding

        DO NOT RETURN THE LOGITS AS THIS TAKES UP ALOT OF MEMORY
        let garbage collector to its work
        logits of shape (batch, sequence, d_vocab) so for a large d_vocab, this is a very big tensor of a few Gbs
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        self.profile_memory("Initial")

        # split into 3 parts along the first dimension
        # first third is chosen, next third is rejected, last third is reasoning
        batch_size = batch["input_ids"].size(0) // 3

        # # this batch will be a dictionary of stuff decided in data_collator.py
        # # dict keys are input_ids, attention_mask, labels
        # # each dict item is a tensor of size (3*batch_size, seq_len)
        # # since (chosen, rejected, reasoning) will give the 3*
        # logger.info_rank0(f"zihe debug batch shape is {batch['labels'].shape}")

        # this may cause cuda memory issues, reduce batch size to alleviate this problem
        # this all logits_is of shape (3 * batch_size, seq_len, d_vocab)

        # keep the data format as is
        all_logits = model(**batch, return_dict=True, use_cache=False).logits

        self.profile_memory("After loading logits")

        # Only convert if you really need it you really need later, not the full 3‑D tensor
        if self.loss_type in {"ipo", "orpo", "simpo"}:  # these need FP32 division later
            all_logits = all_logits.float()

        # # dont need logits here actually
        # # PLEASE REMEMBER TO REMOVE
        # chosen_logits, rejected_logits, reasoning_logits = all_logits.split(batch_size, dim=0)
        # logger.info_rank0(f"zihe debug shape of chosen_logits is {chosen_logits.shape}")
        # logger.info_rank0(f"zihe debug shape of rejected_logits is {rejected_logits.shape}")
        # logger.info_rank0(f"zihe debug shape of reasoning_logits is {reasoning_logits.shape}")

        # logger.info_rank0(f"zihe debug shape of all logits is {all_logits.shape}")

        # # in the original llamafactory code, they upcast to float32 - massive memory consumption!!
        # # our method keeps the data type of bf16 and almost halves the memory

        # this is for labels only
        # all_logps is of shape (batch_size * 3,)
        # valid_length is of shape (batch_size * 3,)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        # here we no longer need all_logits so we free memory for this HUGE tensor
        # saves ALOT of memory!! O3 made this suggestion; I am blown away...
        # but peak memory is still high, memory tends to fluctuate much more

        self.profile_memory("After getting logps")

        del all_logits
        torch.cuda.empty_cache()

        self.profile_memory("After deleting logits")

        # these are of shape (3*batch_size)
        # logger.info_rank0(f"zihe debug shape of all_logps is {all_logps.shape}")
        # logger.info_rank0(f"zihe debug shape of valid_length is {valid_length.shape}")

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # i believe the following is the logits/logps for labels only
        # each of these are of shape (batch_size,)
        chosen_logps, rejected_logps, reasoning_logps = all_logps.split(batch_size, dim=0)
        # logger.info_rank0(f"zihe debug shape of chosen_logps is {chosen_logps.shape}")
        # logger.info_rank0(f"zihe debug shape of rejected_logps is {rejected_logps.shape}")
        # logger.info_rank0(f"zihe debug shape of reasoning_logps is {reasoning_logps.shape}")

        # these lengths are NOT all the same
        # chosen and rejected are the same lengths, but reasoning is a different length
        # these lengths correspond to the labels (not inputs, so excluding the prompt)
        # so these correspond to chosen_labels, rejected_labels, and reasoning_labels
        chosen_length, rejected_length, reasoning_length = valid_length.split(batch_size, dim=0)
        # logger.info_rank0(f"zihe debug lengths of stuff are: {chosen_length},{rejected_length},{reasoning_length}")

        # we can return alot of logps since these take up little memory

        # default loss is sigmoid here
        # last one is a normalized chosen_logps
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            # order of first 2, the chosen and rejected logps must stay the same
            return (
                chosen_logps,
                rejected_logps,
                reasoning_logps,
                chosen_logps,
                rejected_logps,
                reasoning_logps,
            )
        # default is here, will be sigmoid
        # normalize if neccessary
        # these logps take up very little memory so we can return and create many of them
        else:
            return (
                chosen_logps,
                rejected_logps,
                reasoning_logps,
                chosen_logps / chosen_length,
                rejected_logps / rejected_length,
                reasoning_logps / reasoning_length,
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
            policy_chosen_logps_avg,
            policy_rejected_logps_avg,
            policy_reasoning_logps_avg,
        ) = self.concatenated_forward(model, batch)

        self.profile_memory("After concat forward")

        # # logps of shape (batch,)
        # logger.info_rank0(f"zihe check shape of policy chosen logps is{policy_chosen_logps.shape}")
        # logger.info_rank0(f"zihe check shape of policy rejected logps is{policy_rejected_logps.shape}")
        # logger.info_rank0(f"zihe check shape of policy reasoning logps is{policy_reasoning_logps.shape}")
        # # check actual values; whether policy is same as reference
        # # with lora, reference is no lora, policy is with lora
        # logger.info_rank0(f"zihe check full policy chosen logps is{policy_chosen_logps}")

        # Get reference log probs
        # since compute_reference_log_probs calls a method that we override,
        # which is concatenated_forward
        # we must be careful to keep the order of arguments to allow this to work
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)

        self.profile_memory("After computing ref logprobs")

        # logger.info_rank0(f"zihe check shape of reference chosen logps is{reference_chosen_logps.shape}")
        # logger.info_rank0(f"zihe check full reference chosen logps is{reference_chosen_logps}")

        # Compute dpo preference loss
        # this takes in logprobs, which is already averaged over sequence
        dpo_losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        # # # check dpo loss shape, which is simply (batch,)
        # logger.info_rank0(f"zihe check dpo loss shape is {dpo_losses.shape}")

        # not sure why this is included in default DPO implementation for LLaMA-Factory but some form of sft is included here
        # sft on chosen logps
        # default ftx_gamma is zero
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            dpo_losses += self.ftx_gamma * sft_loss

        # For reasoning loss, we simply use the negative log probability
        # This is simpler than the DPO loss formulation because we're not comparing two alternatives
        # We just want to maximize the probability of generating good verification reasoning

        # Combine DPO loss with reasoning loss
        # normalized version
        if self.finetuning_args.norm_reasoning:
            reasoning_loss = -policy_reasoning_logps_avg
        # un-normalized version
        else:
            reasoning_loss = -policy_reasoning_logps

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

        # Add reasoning loss to metrics
        metrics[f"{prefix}reasoning_loss"] = reasoning_loss.mean().detach().item()
        metrics[f"{prefix}dpo_loss"] = dpo_losses.mean().item()
        metrics[f"{prefix}total_loss"] = combined_losses.item()

        # Add special metrics for ORPO
        if self.loss_type == "orpo":
            sft_loss = -policy_chosen_logps_avg
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((dpo_losses - sft_loss) / self.beta).mean().item()

        self.profile_memory("End - computed all losses")

        return combined_losses, metrics

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            # logger.info_rank0("zihe check ref model is None")
            # when finetuning is lora
            # the ref_model is the version without lora; disable lora adapters
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            # logger.info_rank0("zihe check ref model is not None")
            # when finetuning is not lora (full trng)
            # this is the frozen base model
            ref_model = self.ref_model
            ref_context = nullcontext()

        # # if gradient_checkpointing is activated overall, we disable it for the ref model
        # # no need to turn on as ref_model never used for backprop
        # if getattr(ref_model, "gradient_checkpointing", False):
        #     ref_model.gradient_checkpointing_disable()

        # do not use inference mode here as itll cause errors
        with torch.no_grad(), ref_context, torch.cuda.amp.autocast(dtype=torch.bfloat16):
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    # Add this to your CustomDPOTrainer class, perhaps in the get_batch_loss_metrics method
    def profile_memory(self, label=""):
        r"""this is my function to debug whats going on"""
        # needed to sync gpu and cpu for accurate profiling
        torch.cuda.synchronize()
        # check if this is the global main process
        if torch.cuda.is_available() and self.is_world_process_zero():
            print(f"=== Memory Stats ({label}) ===")
            # check which device were using
            # print(f"[rank {torch.dist.get_rank()}]")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            # current memory cached
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            # returns peak allocation since the beginning of the program
            print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
            print("======")

            # More detailed memory snapshot
            print(torch.cuda.memory_summary(abbreviated=True))
            # # for all devices
            # cuda_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            # # as a list of integers
            # cuda_devices = [int(device) for device in cuda_devices_str.split(",")]
            # for dev_id in cuda_devices:
            #     print(torch.cuda.list_gpu_processes(device=dev_id))
