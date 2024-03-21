import os
import fitz
import re
import os
import sys
import spacy
import tiktoken
# import openai
# from langchain.llms import OpenAI
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
# import gradio as gr
import shutil
import tempfile
from tqdm.auto import tqdm
from uuid import uuid4


tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

model_name = "text-embedding-ada-002"
embedder = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
# llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=100, top_p=5,
#                  frequency_penalty=0.0, presence_penalty=0.0)
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="GCP", region="us-central1")


def read_pdf_clean_and_save_text(pdf_path):
    txt_path = pdf_path.replace(".pdf", ".txt")
    print(pdf_path)
    print(txt_path)

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

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def delete_generated_files(pdf_path):
    txt_path = pdf_path.replace(".pdf", ".txt")
    os.remove(txt_path)


def text_loader(file_path):
    txt_path = file_path.replace(".pdf", ".txt")
    loader = TextLoader(txt_path)
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
        print(record_texts)
        print(record_texts[0])
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


def get_vectors_from_pinecone(all_splits):
    index = pc.Index(name="pdf_qa_index")
    # ids = [str(hash(text)) for text in all_splits]
    # response = index.fetch(ids)
    # vectorstore = {k: v["values"] for k, v in response.items()}
    vectorstore = Pinecone(index, embedder.embed_query, all_splits)
    return vectorstore


# def create_chat_model(vectorstore):
#     retriever = vectorstore.as_retriever()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_length=1000)
#     chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
#     return chat


def get_answer_from_chat_model(chat, question):
    dic = {"question": question}
    response = chat(dic)
    answer = response["answer"]
    return answer


def command_running():
    while True:
        print("Please input the path of the pdf file")
        pdf_path = input()
        # pdf_path = r"D:\Iowa State University\2024 Spring\COM S 672\Reviews\2.pdf"
        if not os.path.exists(pdf_path):
            print("The file does not exist")
            continue

        # print("Please input the question")
        # question = str(input())

        read_pdf_clean_and_save_text(pdf_path)
        loader, data = text_loader(pdf_path)
        # create_index(loader)
        # get_all_splits(data)
        ids = vectors_store(data)
        # chat = create_chat_model(ids)
        # answer = get_answer_from_chat_model(chat, question)
        # print(answer)

        print("Do you want to ask another question? (y/n)")
        response = input()
        if response == "n":
            break
        else:
            continue
    # delete_generated_files(pdf_path)


# gradio_input_list = [
#     gr.File(label="Upload your PDF file", type="file"),
#     gr.Textbox(label="Your Question")
# ]
#
# gradio_output_list = [
#     gr.Textbox(label="ChatModel Answer")
# ]


# def process_pdf_and_question(pdf_file, question):
#     temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
#     with open(temp_pdf_path, 'wb') as temp_file:
#         temp_file.write(pdf_file)
#     txt_path = temp_pdf_path.replace(".pdf", ".txt")
#
#     text = read_pdf_clean_and_save_text(temp_pdf_path)
#     loader, data = text_loader(txt_path)
#     create_index(loader)
#     all_splits = get_all_splits(data)
#     ids = vectors_store(all_splits)
#     chat = create_chat_model(ids)
#     answer = get_answer_from_chat_model(chat, question)
#
#     os.remove(temp_pdf_path)
#
#     return answer


# def main():
#     with gr.Blocks() as demo:
#         gr.Markdown("## PDF ChatModel")
#         gr.Markdown("Upload a PDF file and ask a question about its content.")
#
#         with gr.Row():
#             pdf_input = gr.File(label="Upload your PDF file", type="bytes")
#             question_input = gr.Textbox(label="Your Question")
#             submit_button = gr.Button("Submit")
#
#         answer_output = gr.Textbox(label="ChatModel Answer")
#
#         submit_button.click(
#             fn=process_pdf_and_question,
#             inputs=[pdf_input, question_input],
#             outputs=answer_output
#         )
#
#     demo.launch()


if __name__ == "__main__":
    command_running()




