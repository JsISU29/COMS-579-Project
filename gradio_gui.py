import time
import gradio as gr
from preprocess_pdf import upload_index_pdf_to_pinecone
from user_query import get_answer_from_chat_model


top_k = 5
authors_text = """
<div style='font-size: 12px;'>
  <p>Authors: Shuairan Chen, Fan Zhou, Jian Sun</p>
  <p>Upload your PDF files and ask questions about the content.</p>
</div>
"""


def update_top_k(val):
    global top_k
    top_k = val
    return top_k


def slow_echo(message, history):
    answer = get_answer_from_chat_model(message, top_k=top_k)
    print(answer)
    for i in range(len(answer)):
        time.sleep(0.01)
        yield answer[: i + 1]


with gr.Blocks() as demo:
    gr.Markdown("# COM S 579 PDF ChatModel")
    gr.Markdown(authors_text)

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
        with gr.Column():
            top_k_slider = gr.Slider(label="Top K", minimum=5, maximum=10, step=1)
            update = gr.Button("Update Top K")
            update.click(
                fn=update_top_k,
                inputs=top_k_slider
            )
        chat_box = gr.ChatInterface(fn=slow_echo).queue()

    top_k_slider.release(update_top_k, inputs=top_k_slider)


if __name__ == "__main__":
    demo.launch()
