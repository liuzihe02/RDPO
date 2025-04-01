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

from llamafactory.data.mm_plugin import VideoInput

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen
from .pairwise import PairwiseDatasetProcessor
from typing_extensions import override

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


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

            # check how long is response and prompt
            print(f" zihe prompt is {examples['_prompt'][i]}")
            print(f" zihe response is {examples['_response'][i]}")

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

    @override
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: str | None,
        tools: str | None,
        images: Sequence[str | bytes | EncodedImage | Image],
        videos: Sequence[str],
        audios: Sequence[str | ndarray[Any, dtype]],
    ) -> Tuple[List[int]]:
        return super()._encode_data_example(prompt, response, system, tools, images, videos, audios)
