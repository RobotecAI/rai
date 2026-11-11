# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_benchmark_llm_from_vendor_and_name(calls):
    from rai_bench.utils import get_llm_for_benchmark

    get_llm_for_benchmark("llama3.2", "ollama")
    get_llm_for_benchmark("gemini-3-pro", "google", temperature=0)
    assert calls == [
        ("llama3.2", {"model_provider": "ollama", "keep_alive": 20}),
        ("gemini-3-pro", {"model_provider": "google_genai", "temperature": 0}),
    ]
