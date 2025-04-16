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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen

# wrap around this
from .pairwise import PairwiseDatasetProcessor
from typing_extensions import override


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class RDPOPairwiseDatasetProcessor(PairwiseDatasetProcessor):
    @override
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
        # tuple of 6 lists
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
        """
        Extended version of the PairwiseDatasetProcessor's _encode_data_example method
        that also processes reasoning data from responses.

        Expects response to contain:
        - response[0]: chosen response
        - response[1]: rejected response
        - response[2]: reasoning (if available)

        Returns a tuple containing the original pairwise data plus reasoning information.
        """
        # Call the parent class method to get the base pairwise data, for chosen and rejected
        # this standardizes length of chosen and rejected to be the same!
        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = super()._encode_data_example(
            prompt, response[:2], system, tools, images, videos, audios
        )

        # Process reasoning
        # ids here refer to prompt + reasoning
        # labels is just the prompt itself
        reasoning_input_ids, reasoning_labels = [], []
        # Process reasoning message
        reasoning_message = self.template.mm_plugin.process_messages(
            prompt + [response[2]],  # Use the reasoning as the response
            images,
            videos,
            audios,
            self.processor,
        )

        # Encode reasoning
        prompt_ids, reasoning_ids = self.template.encode_oneturn(self.tokenizer, reasoning_message, system, tools)

        if self.template.efficient_eos:
            reasoning_ids += [self.tokenizer.eos_token_id]

        # Process token IDs for multimodal inputs
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # Apply length constraints
        # zihe TODO: the way the len(reasoning_ids) instead of using max may cause problems but hopefully its fine if the cutoff length is long enough
        # source length is max length of prompt
        # target length is max length of reasoning
        source_len, target_len = infer_seqlen(len(prompt_ids), len(reasoning_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        reasoning_ids = reasoning_ids[:target_len]

        # Create input and label sequences
        reasoning_input_ids = prompt_ids + reasoning_ids
        reasoning_labels = [IGNORE_INDEX] * source_len + reasoning_ids

        # Return standard pairwise data along with reasoning data
        return (
            chosen_input_ids,
            chosen_labels,
            rejected_input_ids,
            rejected_labels,
            reasoning_input_ids,
            reasoning_labels,
        )

    # override the parent pairwise dataset
    @override
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # Process the example with reasoning
            results = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            # Unpack results
            (
                chosen_input_ids,
                chosen_labels,
                rejected_input_ids,
                rejected_labels,
                reasoning_input_ids,
                reasoning_labels,
            ) = results

            # Add to model inputs
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["reasoning_input_ids"].append(reasoning_input_ids)
            model_inputs["reasoning_attention_mask"].append([1] * len(reasoning_input_ids))
            model_inputs["reasoning_labels"].append(reasoning_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    @override
    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        """Print information about a data example, including reasoning data."""
        # Print standard pairwise data
        super().print_data_example(example)

        # Print reasoning data
        valid_reasoning_labels = list(filter(lambda x: x != IGNORE_INDEX, example["reasoning_labels"]))
        print("reasoning_input_ids:\n{}".format(example["reasoning_input_ids"]))
        print(
            "reasoning_inputs:\n{}".format(
                self.tokenizer.decode(example["reasoning_input_ids"], skip_special_tokens=False)
            )
        )
        print("reasoning_label_ids:\n{}".format(example["reasoning_labels"]))
        print(f"reasoning_labels:\n{self.tokenizer.decode(valid_reasoning_labels, skip_special_tokens=False)}")
