import argparse

from ament_index_python.packages import get_package_share_directory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rai.apps.talk_to_docs import ingest_documentation


def main():
    parser = argparse.ArgumentParser(description="Load documentation into FAISS")
    parser.add_argument(
        "documentation_root", type=str, help="Path to the root of the documentation"
    )
    args = parser.parse_args()

    docs = ingest_documentation(documentation_root=args.documentation_root)
    faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())
    faiss_index.add_documents(docs)
    save_dir = get_package_share_directory("rai_whoami")
    faiss_index.save_local(save_dir)


if __name__ == "__main__":
    main()
