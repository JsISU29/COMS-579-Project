import time
import gradio as gr
from preprocess_pdf import upload_index_pdf_to_pinecone
from user_query import get_answer_from_chat_model


def slow_echo(message, history):
    answer = get_answer_from_chat_model(message)
    print(answer)
    for i in range(len(answer)):
        time.sleep(0.01)
        yield answer[: i + 1]


with gr.Blocks() as demo:
    gr.Markdown("# COM S 579 PDF ChatModel")
    gr.Markdown("# Authors: Shuairan Chen, Fan Zhou, Jian Sun")
    gr.Markdown("# Upload your PDF files and ask questions about the content.")

    with gr.Tab("Upload PDF"):
        gr.Markdown("# Upload and Indexing PDF file")
        pdf_file = gr.File(label="Upload your PDF file", file_types=["pdf"], type="filepath")
        output = gr.Textbox(label="Upload Result", type="text")
        button = gr.Button("Upload PDF")
        button.click(
            fn=upload_index_pdf_to_pinecone,
            inputs=pdf_file,
            outputs=output
        )

    with gr.Tab("Chat with Model"):
        gr.Markdown("# Chat with the PDF Chat Bot")
        chat_box = gr.ChatInterface(fn=slow_echo).queue()


if __name__ == "__main__":
    demo.launch()
