# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from ..dpo.trainer import CustomDPOTrainer
from ...extras.constants import IGNORE_INDEX


class RDPOTrainer(CustomDPOTrainer):
    """
    Trainer for Reasoning-based Direct Preference Optimization (RDPO).

    This trainer extends DPO to incorporate reasoning behind preferences
    """

    def tokenize_row(self, feature, model=None) -> Dict:
        """
        Tokenize a single row from a RDPO specific dataset.
        Extends the DPOTrainer.tokenize_row method to also tokenize the reasoning field.
        """
        # Tokenize standard fields (prompt, chosen, rejected)
        result = super().tokenize_row(feature, model)

        # Tokenize reasoning field if it exists
        if "reasoning" in feature and feature["reasoning"] is not None:
            if self.is_encoder_decoder:
                reasoning_tokens = self.tokenizer(
                    feature["reasoning"], truncation=True, max_length=self.max_target_length, add_special_tokens=True
                )
                result["reasoning_labels"] = reasoning_tokens["input_ids"]
            else:
                if not isinstance(feature["reasoning"], str):
                    raise ValueError(f"reasoning should be an str but got {type(feature['reasoning'])}")

                if self.is_vision_model:
                    reasoning_tokens = self.processor(
                        feature["reasoning"], images=feature.get("images"), add_special_tokens=False
                    )
                    reasoning_tokens = {k: v[0] for k, v in reasoning_tokens.items()}
                else:
                    reasoning_tokens = self.tokenizer(feature["reasoning"], add_special_tokens=False)

                # Add BOS token to head of reasoning if not already there
                bos_token_id = self.tokenizer.bos_token_id
                if len(reasoning_tokens["input_ids"]) == 0 or bos_token_id != reasoning_tokens["input_ids"][0]:
                    reasoning_tokens["input_ids"] = [bos_token_id] + reasoning_tokens["input_ids"]
                    reasoning_tokens["attention_mask"] = [1] + reasoning_tokens["attention_mask"]

                # Add EOS token to end of reasoning if not already there
                eos_token_id = self.tokenizer.eos_token_id
                if len(reasoning_tokens["input_ids"]) == 0 or eos_token_id != reasoning_tokens["input_ids"][-1]:
                    reasoning_tokens["input_ids"].append(eos_token_id)
                    reasoning_tokens["attention_mask"].append(1)

                # Create labels (same as input_ids for causal language modeling)
                reasoning_labels = reasoning_tokens["input_ids"][:]

                # Add reasoning tokens to result
                result["reasoning_input_ids"] = reasoning_tokens["input_ids"]
                result["reasoning_attention_mask"] = reasoning_tokens["attention_mask"]
                result["reasoning_labels"] = reasoning_labels

        return result

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Computes the RDPO loss and other metrics for the given batch of inputs.

        RDPO loss = DPO loss - reasoning_weight * log p(reasoning)
        """
        # Get standard DPO loss and metrics
        dpo_loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        # Compute reasoning loss if reasoning is provided
        if "reasoning_input_ids" in batch and "reasoning_attention_mask" in batch:
            # Prepare reasoning inputs
            reasoning_inputs = {
                "input_ids": batch["reasoning_input_ids"],
                "attention_mask": batch["reasoning_attention_mask"],
                "labels": batch["reasoning_labels"],
            }
            reasoning_inputs = self._prepare_inputs(reasoning_inputs)

            # Compute loss for reasoning generation
            with self.compute_loss_context_manager():
                reasoning_outputs = model(**reasoning_inputs)
                reasoning_loss = reasoning_outputs.loss

            # Apply weight to reasoning component
            # Default to 1.0 if not specified
            reasoning_weight = getattr(self.finetuning_args, "rdpo_reasoning_weight", 1.0)

            # Total loss combines DPO loss with the negative log probability of the reasoning
            # We want to maximize log p(reasoning), so we subtract reasoning_loss
            total_loss = dpo_loss + reasoning_weight * reasoning_loss

            # Update metrics
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}loss/dpo"] = dpo_loss.detach().cpu()
            metrics[f"{prefix}loss/reasoning"] = reasoning_loss.detach().cpu()

            return total_loss, metrics

        return dpo_loss, metrics
