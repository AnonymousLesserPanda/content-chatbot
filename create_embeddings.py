import argparse
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding website content")

    parser.add_argument(
        "-z",
        "--zendesk",
        type=str,
        required=False,
        default="https://support.strikingly.com/api/v2/help_center/en-us/articles.json",
        help="URL to your zendesk api",
    )
    args = parser.parse_args()

    r = requests.get(args.zendesk)
    articles = r.json().get("articles", [])
    pages = [
        {"text": clean_html(article["body"]), "source": article["html_url"]}
        for article in articles
    ]

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page["text"])
        docs.extend(splits)
        metadatas.extend([{"source": page["source"]}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(
        docs,
        AzureOpenAIEmbeddings(
            openai_api_key="3TYDIcyQ3pFOZV8hJdhjajPXfTapwF8e0sCSg580Cetrhic9OhnTJQQJ99BBACHYHv6XJ3w3AAAAACOGYmHV",
            azure_endpoint="https://ai-redpandadeployment385863590335.openai.azure.com/",
            deployment="text-embedding-3-large",
            openai_api_version="2023-05-15",
        ),
        metadatas=metadatas,
    )

    store.save_local("faiss_store")
