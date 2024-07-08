import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Type

import tqdm
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_FILETYPES: List[Annotated[str, 'filetype e.g. ".txt"']] = [
    ".txt",
    ".pdf",
    ".docx",
]

DEFAULT_PARSERS: Dict[Annotated[str, 'filetype e.g. ".txt."'], Type[BaseLoader]] = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(
    paths: List[Path], parsers: Optional[Dict[str, Type[BaseLoader]]] = None
) -> List[Document]:
    documents: List[Document] = []

    if parsers is None:
        parsers = DEFAULT_PARSERS

    for path in tqdm.tqdm(paths, desc="Loading documents"):
        document = parsers[path.suffix](file_path=path).load_and_split()  # type: ignore
        documents.extend(document)

    return documents


def ingest_documentation(
    documentation_root: Path, recursive: bool = True
) -> List[Document]:
    documents = find_documents(documentation_root, recursive=recursive)
    return load_documents(documents)


def find_documents(
    path: Path | str,
    filetypes: Optional[List[str]] = None,
    recursive: bool = True,
    verbose: bool = False,
) -> List[Path]:
    if filetypes is None:
        filetypes = DEFAULT_SUPPORTED_FILETYPES
    documents: List[Path] = []

    path = Path(path)
    finder_function = path.rglob if recursive else path.glob
    for file in finder_function("*"):
        if verbose:
            logger.info(f"Checking {file}")
        if file.is_file() and file.suffix in filetypes:
            documents.append(file)
    return documents
