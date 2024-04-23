import os
import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import numpy as np

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


def get_document_by_id(doc_id):
    index = get_exists_index_from_pinecone(index_name)
    response = index.fetch(ids=[doc_id])
    if 'vectors' in response and doc_id in response['vectors']:
        document_data = response['vectors'][doc_id]
        if 'metadata' in document_data and 'text' in document_data['metadata']:
            document_path = document_data['metadata']['source']
            document_name = os.path.basename(document_path)
            doc_dict = {"document_name": document_name, "text": document_data['metadata']['text']}
            return doc_dict
    return None


def retrieve_documents_from_pinecone(index_name, query_vector, top_k=5):
    pinecone_index = get_exists_index_from_pinecone(index_name)
    if pinecone_index is None:
        raise ValueError(f"No index found with the name {index_name}")
    # results = pinecone_index.query(queries=query_vector, top_k=top_k)
    results = pinecone_index.query(vector=query_vector, top_k=top_k)

    retrieved_doc_ids = [match["id"] for match in results.get('matches', [])]
    retrieved_scores = {match["id"]: match["score"] for match in
                        results.get('matches', [])}
    retrieved_docs = [get_document_by_id(doc_id) for doc_id in retrieved_doc_ids]

    return retrieved_docs, retrieved_scores


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def rerank_documents(query_vector, documents, top_p=5):
    similarities = []
    for doc in documents:
        doc_vector = embedder.embed_documents(doc["text"])
        if isinstance(doc_vector, list):
            doc_vector = np.array(doc_vector)

        if isinstance(doc_vector, np.ndarray) and doc_vector.ndim == 2:
            doc_vector = np.mean(doc_vector, axis=0)  # Average the embeddings of the document

        if isinstance(doc_vector, list):
            doc_vector = np.array(doc_vector)

        similarity = cosine_similarity(query_vector, doc_vector)
        doc["similarity"] = similarity

    sorted_data = sorted(documents, key=lambda x: x['similarity'], reverse=True)
    top_p = min(top_p, len(documents))
    top_p_documents = [sorted_data[i] for i in range(top_p)]

    return top_p_documents


def get_aggregated_docs(query_vector, top_k=5, top_p=5):
    retrieved_docs, retrieved_scores = retrieve_documents_from_pinecone(index_name, query_vector, top_k=top_k)
    ranked_docs = rerank_documents(query_vector, retrieved_docs, top_p=top_p)

    referencing = ""
    aggregated_docs = ""
    for i in range(0, top_p):
        doc = ranked_docs[i]
        doc_name = doc["document_name"]
        similarity = doc["similarity"]
        titles = f"{i + 1}. From document {doc_name} (similarity={similarity}):\n"
        text = doc["text"].replace("\n", " ")
        referencing += f"{titles} {text}\n"
        aggregated_docs += f"{text} "

    return aggregated_docs, referencing


def create_chat_model(index, top_k=5):
    text_field = "text"
    vectorstore = PineconeVectorStore(index, embedder, text_field)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_length=1000)
    model_kwargs = {
        "top_p": 0.9,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0
    }
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1000,
                     model_kwargs=model_kwargs)
    chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return chat


def normalized_query_vector(query_vector):
    query_vector = np.array(query_vector, dtype=np.float32)

    norm = np.linalg.norm(query_vector)
    if norm == 0:
        norm = 1

    normalize_query_vector = query_vector / norm
    return normalize_query_vector.tolist()


def get_answer_from_chat_model(question, top_k=5):
    index = get_exists_index_from_pinecone(index_name)
    chat = create_chat_model(index, top_k)
    query_vector = embedder.embed_query(question)
    reference = ""

    normalize_query_vector = normalized_query_vector(query_vector)
    # print(normalize_query_vector)
    aggregated_docs, refer = get_aggregated_docs(normalize_query_vector, top_k=top_k)
    reference = "Reference:\n" + refer
    combined_input = f"Question: {question}\n\n{reference}"
    dic = {
        'question': combined_input
    }

    response = chat.invoke(dic)
    answer = response["answer"]
    answer += "\n\n" + reference
    return answer


def command_running():
    while True:
        print("Please input your question")
        print("User: ", end="")
        question = str(input())
        print("Please input your question top_k value (5-10):", end="")
        top_k = int(input())

        answer = get_answer_from_chat_model(question, top_k)
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
