# COMS-579-Project
579 final project

# Team Members:

* Shuairan Chen

* Fan Zhou

* Jian Sun

# Retrieval-augmented generation (RAG) 

Retrieval-augmented generation (RAG) is a framework to give generative models knowledge without finetuning themselves. In this way, an LLM can adapt to new tasks quickly with the presence of new documents.

# Features

Langchain Pipeline: Utilizes cutting-edge language processing models to analyze, generate, and manipulate text data.

Pinecone Vector Database: Efficiently handles high-dimensional vector data, enabling fast and scalable similarity search.

Gradio Interface: Provides an easy-to-use web interface for users to interact with the application, input data, and visualize results.

# Prerequisites

Python 3.10 or higher

An OpenAI API key in your System Evironment for accessing OpenAI's embeddings.
You could set up your OpenAI API key in system enviroment by: https://platform.openai.com/docs/quickstart?context=python

A Pinecone API key in your System Evironment for indexing and querying vectors.
You could set up your Pinecone API key in system enviroment by: https://docs.pinecone.io/guides/getting-started/authentication

# Installation

1. Clone the repository:

`git clone https://github.com/JsISU29/COMS-579-Project.git`

2. Navigate to the project directory:

`cd COMS-579-Project`

3. For MacOS need Xcode Tool to compile some python library, run the following command in terminal, then follow the window which shows after you run the command to install Xcode:
`xcode-select --install`

4. Install the required Python packages:
`pip install --upgrade pip`

`pip install -r requirements.txt`

If still show no some modules, please try the following commands:

`pip install --upgrade PyMuPDF`

`pip install --upgrade tiktoken`

`pip install --upgrade pinecone-client`

`pip install --upgrade tqdm`

`pip install --upgrade uuid`

`pip install --upgrade langchain`

`pip install --upgrade langchain-core`

`pip install --upgrade langchain_community`

`pip install --upgrade langchain_openai`

`pip install --upgrade langchain-text-splitters`

`pip install --upgrade langchain_pinecone`

`pip install --upgrade pypdf`

`pip install --upgrade gradio`

`pip install --upgrade gradio_client`

`pip install --upgrade numpy`

`pip install --upgrade fitz`

# Usage

Project Milestone 1 - Upload and Indexing PDF

1. To run the project for now, please execute the main script:

`python preprocess_pdf.py`

or use an IDE such as PyCharm to run the preprocess_pdf.py file

2. Then need you to enter your pdf file path to save the pdf content into PineCone DB

Project Milestone 2 - Answer User Queries

1. To run the project for now, please execute the main script:

`python user_query.py`

or use an IDE such as PyCharm to run the user_query.py file

2. Then need you to enter your question to get the answer from the chat model

Project Milestone 3 - The GUI App

1. To run the project for now, please execute the main script:

`python gradio_gui.py`

or or use an IDE such as PyCharm to run the gradio_gui.py file

2. Click the "Unload PDF" tag to upload your pdf, you could drag your pdf file to the upload box

3. Then click the bottom Upload PDF button to start upload pdf to PineCone DB, if upload success, the upload result box while show success

4. Click the "Chat with Model" tag to talk with the chat model

5. You could change the top-k value by the Top K Slider, after changed the value you need to click the Update Top K button

6. Then you could input your query in the bottom text box, then press "Enter" or click the Submit button to send your query and get the answer

7. Then the query and the results while show in the Chatbot Box

# Demo Video

You could find the demo video about Upload and Indexing PDF in the repo which named "COMS579ProjectFirstDemo.mp4", this video need HEVC, if your player not support the video, you could use the "COMS579ProjectFirstDemo-Fix-NoNeedHEVC.mp4" to watch the demo video.

You could find the demo video about Answer User Queries in the repo which named "COMS579ProjectSecondDemo.mp4", this video need AV1, if your player not support the video, you could use the "COMS579ProjectSecondDemo-Fix-NoNeedAV1.mp4" to watch the demo video.

You could find the demo video about the Gradio GUI in the repo which named "COMS579ProjectThirdDemo.mp4"

# Acknowledgments

The creators of Langchain for their innovative approach to language processing.

Pinecone team for providing an efficient vector database solution.

Gradio for making it easy to create web interfaces for machine learning models.
