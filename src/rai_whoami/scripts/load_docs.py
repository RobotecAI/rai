import argparse
import glob

from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rai.apps.talk_to_docs import ingest_documentation
from rai.scenario_engine.messages import HumanMultimodalMessage, preprocess_image


def main():
    parser = argparse.ArgumentParser(description="Load documentation into FAISS")
    parser.add_argument(
        "documentation_root", type=str, help="Path to the root of the documentation"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output directory",
    )
    args = parser.parse_args()

    docs = ingest_documentation(
        documentation_root=args.documentation_root + "/documentation"
    )
    faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())
    faiss_index.add_documents(docs)
    save_dir = args.output
    faiss_index.save_local(save_dir)

    prompt = (
        "You will be given a robot's documentation. "
        "Your task is to identify the robot's identity. "
        "The description should cover the most important aspects of the robot with respect to human interaction, "
        "as well as the robot's capabilities and limitations including sensor and actuator information. "
        "If there are any images provided, make sure to take them into account by thoroughly analyzing them. "
        "Your reply should start with I am a ..."
    )
    llm = ChatOpenAI(model="gpt-4o-mini")

    images = glob.glob(args.documentation_root + "/images/*")

    messages = [SystemMessage(content=prompt)] + [
        HumanMultimodalMessage(
            content=str([doc.page_content for doc in docs]),
            images=[preprocess_image(image) for image in images],
        )
    ]
    output = llm.invoke(messages)
    with open(save_dir + "/robot_identity.txt", "w") as f:
        f.write(output.content)


if __name__ == "__main__":
    main()
