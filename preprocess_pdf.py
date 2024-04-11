import fitz
import re
import os
import tiktoken
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from tqdm.auto import tqdm
from uuid import uuid4

tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "coms579pdfqa"
model_name = "text-embedding-ada-002"
embedder = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="GCP", region="us-central1")


def read_pdf_clean_and_save_text(pdf_path):
    # txt_path = pdf_path.replace(".pdf", ".txt")
    # print(pdf_path)
    # print(txt_path)

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()

    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove multiple new lines
    text = re.sub(r'\n+', '\n', text)

    # os.remove(pdf_path)

    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     f.write(text)
    return text


def delete_generated_files(pdf_path):
    txt_path = pdf_path.replace(".pdf", ".txt")
    os.remove(txt_path)


def text_loader(file_path):
    txt_path = file_path.replace(".pdf", ".txt")
    loader = TextLoader(txt_path)
    data = loader.load()
    return loader, data


def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    return loader, data


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def get_all_splits(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=tiktoken_len,
                                                   separators=["\n\n", "\n", " ", ""])
    all_splits = text_splitter.split_documents(data)
    return all_splits


def create_indexing(data, index):
    batch_limit = 100
    texts = []
    meta_datas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=tiktoken_len,
                                                   separators=["\n\n", "\n", " ", ""])

    for i, document in enumerate(tqdm(data)):
        metadata = document.metadata
        record_texts = text_splitter.split_text(document.page_content)
        record_metadata = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
        texts.extend(record_texts)
        meta_datas.extend(record_metadata)

        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeddings = embedder.embed_documents(texts)
            vectors = list(zip(ids, embeddings, meta_datas[:len(texts)]))
            index.upsert(vectors=vectors)
            texts = []
            meta_datas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = embedder.embed_documents(texts)
        vectors = list(zip(ids, embeddings, meta_datas[:len(texts)]))
        index.upsert(vectors=vectors)


def vectors_store(data):
    index_name = "coms579pdfqa"
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    if index_name not in existing_indexes:
        pc.create_index(name=index_name, dimension=1536, metric="dotproduct", spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(name=index_name)
    time.sleep(1)

    create_indexing(data, index)

    return index


def upload_index_pdf_to_pinecone(file_path):
    if not os.path.exists(file_path):
        return "The file does not exist"
    loader, data = pdf_loader(file_path)
    vectors_store(data)
    return "Upload index to Pinecone success"


def command_running():
    while True:
        print("Please input the path of the pdf file")
        pdf_path = input()

        if not os.path.exists(pdf_path):
            print("The file does not exist")
            continue

        loader, data = pdf_loader(pdf_path)
        ids = vectors_store(data)
        print("Upload index to Pinecone success")

        print("Do you want to upload another pdf file? (y/n)")
        response = input()
        if response == "n":
            break
        else:
            continue

    print("Upload PDF files finished.")


if __name__ == "__main__":
    command_running()







