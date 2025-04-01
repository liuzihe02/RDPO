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
from .pairwise import PairwiseDatasetProcessor
from typing_extensions import override


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class RDPOPairwiseDatasetProcessor(PairwiseDatasetProcessor):
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

            # process data using parent class method
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            # Add rationale processing
            rationale = examples.get("_reasoning", [None])[i]
            if rationale:
                rationale_ids = self.tokenizer.encode(rationale, add_special_tokens=False)
                if len(rationale_ids) > 0:
                    # Truncate if needed
                    rationale_ids = rationale_ids[: min(len(rationale_ids), 512)]
                    model_inputs["rationale_ids"].append(rationale_ids)
                else:
                    model_inputs["rationale_ids"].append([])
            else:
                model_inputs["rationale_ids"].append([])

            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
