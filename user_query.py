import os
import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "coms579pdfqa"
model_name = "text-embedding-ada-002"
embedder = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="GCP", region="us-central1")


def get_exists_index_from_pinecone(index_name):
    # ids = [str(hash(text)) for text in all_splits]
    # response = index.fetch(ids)
    # vectorstore = {k: v["values"] for k, v in response.items()}
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        # print(f"Get index of {index_name} from Pinecone success")
        return pc.Index(name=index_name)

    # print(f"Get index of {index_name} from Pinecone failed")
    return None


def create_chat_model(index):
    text_field = "text"
    vectorstore = PineconeVectorStore(index, embedder, text_field)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_length=1000)
    model_kwargs = {
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1000, model_kwargs=model_kwargs)
    chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return chat


def get_answer_from_chat_model(question):
    index = get_exists_index_from_pinecone(index_name)
    chat = create_chat_model(index)
    dic = {
        'question': question,
    }
    response = chat.invoke(dic)
    answer = response["answer"]
    return answer


def command_running():
    while True:
        print("Please input your question")
        print("User: ", end="")
        question = str(input())
        answer = get_answer_from_chat_model(question)
        print("Chatbot:", answer)

        print("Do you want to ask another question? (y/n)")
        response = input()
        if response == "n":
            break
        else:
            continue

    print("Chatbot conversation is closed.")


if __name__ == "__main__":
    command_running()
