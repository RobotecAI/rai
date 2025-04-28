# Copyright (C) 2025 Robotec.AI
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

import argparse
from pathlib import Path

from rai_whoami import EmbodimentSource
from rai_whoami.pipeline import PipelineBuilder
from rai_whoami.processors import (
    CompressorPostProcessor,
    DocsPreProcessor,
    ImagePreProcessor,
    StylePostProcessor,
)
from rai_whoami.vector_db import FAISSBuilder


def build_whoami(args: argparse.Namespace) -> None:
    builder = PipelineBuilder()
    builder.add_preprocessor(DocsPreProcessor())
    builder.add_preprocessor(ImagePreProcessor())
    if args.compress:
        builder.add_postprocessor(CompressorPostProcessor())
    if args.style:
        builder.add_postprocessor(StylePostProcessor())
    if args.build_vector_db:
        builder.add_vector_db_builder(FAISSBuilder(args.output_dir))
    pipeline = builder.build()

    source = EmbodimentSource.from_directory(args.documentation_dir)
    info = pipeline.process(source)
    info.to_directory(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("documentation_dir", type=Path)
    parser.add_argument("--build-vector-db", default=False, action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--compress", default=False, action="store_true")
    parser.add_argument("--style", default=False, action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.documentation_dir / "generated"
    build_whoami(args)
