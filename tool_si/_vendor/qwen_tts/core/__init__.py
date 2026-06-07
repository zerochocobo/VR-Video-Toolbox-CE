# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
__all__ = [
    "Qwen3TTSTokenizerV1Config",
    "Qwen3TTSTokenizerV1Model",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2Model",
]


def __getattr__(name):
    if name == "Qwen3TTSTokenizerV1Config":
        from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config

        return Qwen3TTSTokenizerV1Config
    if name == "Qwen3TTSTokenizerV1Model":
        from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model

        return Qwen3TTSTokenizerV1Model
    if name == "Qwen3TTSTokenizerV2Config":
        from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config

        return Qwen3TTSTokenizerV2Config
    if name == "Qwen3TTSTokenizerV2Model":
        from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

        return Qwen3TTSTokenizerV2Model
    raise AttributeError(name)
