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

import base64
import glob
import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from PIL import Image
from pydantic import BaseModel, Field
from rai.messages import preprocess_image
from rai.messages.multimodal import SystemMultimodalMessage


class EmbodimentInfoDirectoryStructure(Enum):
    RULES = "rules.txt"
    CAPABILITIES = "capabilities.txt"
    BEHAVIORS = "behaviors.txt"
    DESCRIPTION = "description.txt"
    IMAGES = "images"


class DocumentLoader(Enum):
    PDF = PyPDFLoader
    DOCX = Docx2txtLoader
    TXT = TextLoader


class EmbodimentSourceDirectoryStructure(Enum):
    DOCUMENTATION = Path("./documentation")
    IMAGES = Path("./images")
    URDFS = Path("./urdfs")


ALLOWED_EXTENSIONS = {
    ".pdf": DocumentLoader.PDF,
    ".docx": DocumentLoader.DOCX,
    ".doc": DocumentLoader.DOCX,
    ".txt": DocumentLoader.TXT,
    ".md": DocumentLoader.TXT,
    ".urdf": DocumentLoader.TXT,
    ".xacro": DocumentLoader.TXT,
}


class EmbodimentSourceLoader:
    def __init__(
        self,
        root_dir: str | Path,
        extension_to_loader: Optional[Dict[str, "DocumentLoader"]] = None,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.logger = logging.getLogger(__name__)
        self.extension_to_loader = extension_to_loader or ALLOWED_EXTENSIONS

    def load(self) -> "EmbodimentSource":
        return EmbodimentSource(
            documentation=self.load_documentation(),
            images=self.load_images(),
            urdfs=self.load_urdfs(),
        )

    def load_documentation(self) -> List[Document]:
        extension_to_paths: Dict[str, List[str]] = {}
        for extension in self.extension_to_loader:
            extension_to_paths[extension] = glob.glob(
                os.path.join(
                    self.root_dir
                    / EmbodimentSourceDirectoryStructure.DOCUMENTATION.value,
                    "**/*" + extension,
                ),
                recursive=True,
            )
        documents: List[Document] = []
        for extension, files in extension_to_paths.items():
            for file in files:
                if extension not in self.extension_to_loader:
                    self.logger.warning(
                        f"Skipping file {file} because it has an unsupported extension {extension}"
                    )
                    continue
                loader = self.extension_to_loader[extension].value(file_path=file)
                documents.extend(loader.load())
        return documents

    def load_images(self) -> List[str]:
        files = glob.glob(
            os.path.join(
                self.root_dir / EmbodimentSourceDirectoryStructure.IMAGES.value,
                "**/*",
            ),
            recursive=True,
        )
        image_files = [
            file
            for file in files
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")
        ]
        return [preprocess_image(Image.open(file)) for file in image_files]

    def load_urdfs(self) -> List[Document]:
        files = glob.glob(
            os.path.join(
                self.root_dir / EmbodimentSourceDirectoryStructure.URDFS.value,
                "**/*",
            ),
            recursive=True,
        )
        urdf_files = [
            Path(file)
            for file in files
            if file.endswith(".urdf") or file.endswith(".xacro")
        ]
        documents: List[Document] = []
        for file in urdf_files:
            loader = self.extension_to_loader[file.suffix].value(file_path=file)
            documents.extend(loader.load())
        return documents


class EmbodimentSource(BaseModel):
    documentation: List[Document]
    images: List[str] = Field(..., exclude=True)
    urdfs: List[Document]

    @classmethod
    def from_directory(cls, directory: Path | str) -> "EmbodimentSource":
        loader = EmbodimentSourceLoader(directory)
        return loader.load()


class EmbodimentInfo(BaseModel):
    rules: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    behaviors: Optional[List[str]] = None
    description: Optional[str] = None
    images: Optional[Annotated[List[str], "base64 encodedpng images"]] = None

    @classmethod
    def from_file(cls, file: Path | str) -> "EmbodimentInfo":
        if isinstance(file, str):
            file = Path(file)
        return cls.model_validate_json(file.read_text())

    @classmethod
    def from_directory(cls, directory: Path | str) -> "EmbodimentInfo":
        if isinstance(directory, str):
            directory = Path(directory)
        return cls.model_validate_json(
            (directory / "generated" / "info.json").read_text()
        )

    def to_directory(self, directory: Path | str):
        if isinstance(directory, str):
            directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)
        with open(directory / "info.json", "w") as f:
            f.write(self.model_dump_json(indent=4))
        if self.images is not None:
            for image in self.images:
                with open(directory / f"{uuid.uuid4()}.png", "wb") as f:
                    f.write(base64.b64decode(image))

    def to_langchain(
        self, style: Literal["xml_tags", "markdown"] = "xml_tags"
    ) -> SystemMultimodalMessage:
        if style == "xml_tags":
            content = self._to_xml()
        elif style == "markdown":
            content = self._to_markdown()
        else:
            raise ValueError(f"Invalid style: {style}")
        return SystemMultimodalMessage(content=content, images=self.images)

    def _to_xml(self) -> str:
        content = f"<description>\n{self.description}\n</description>\n"
        if self.rules is not None:
            rules = "\n".join(self.rules)
            content += f"<rules>\n{rules}\n</rules>\n"
        if self.capabilities is not None:
            capabilities = "\n".join(self.capabilities)
            content += f"<capabilities>\n{capabilities}\n</capabilities>\n"
        if self.behaviors is not None:
            behaviors = "\n".join(self.behaviors)
            content += f"<behaviors>\n{behaviors}\n</behaviors>\n"
        return content

    def _to_markdown(self) -> str:
        content = f"# Description\n{self.description}\n"
        if self.rules is not None:
            rules = "\n".join(self.rules)
            content += f"# Rules\n{rules}\n"
        if self.capabilities is not None:
            capabilities = "\n".join(self.capabilities)
            content += f"# Capabilities\n{capabilities}\n"
        if self.behaviors is not None:
            behaviors = "\n".join(self.behaviors)
            content += f"# Behaviors\n{behaviors}\n"
        return content

    def __add__(self, other: "EmbodimentInfo") -> "EmbodimentInfo":
        rules = (self.rules or []) + (other.rules or [])
        capabilities = (self.capabilities or []) + (other.capabilities or [])
        behaviors = (self.behaviors or []) + (other.behaviors or [])
        description = (self.description or "") + (other.description or "")
        images = (self.images or []) + (other.images or [])
        return EmbodimentInfo(
            rules=rules,
            capabilities=capabilities,
            behaviors=behaviors,
            description=description,
            images=images,
        )
