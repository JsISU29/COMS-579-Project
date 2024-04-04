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

An OpenAI API key in your System Evironment for accessing OpenAI's embeddings
You could set up your OpenAI API key in system enviroment by: https://platform.openai.com/docs/quickstart?context=python

A Pinecone API key in your System Evironment for indexing and querying vectors
You could set up your Pinecone API key in system enviroment by: https://docs.pinecone.io/guides/getting-started/authentication

# Installation

1. Clone the repository:

`git clone https://github.com/JsISU29/COMS-579-Project.git`

2. Navigate to the project directory:

`cd COMS-579-Project`

3. Install the required Python packages:

`pip install -r requirements.txt`

# Usage

1. To run the project for now, please execute the main script:

`python preprocess_pdf.py`

or use an IDE such as PyCharm to run the preprocess_pdf.py file

2. Then need you to enter your pdf file path to save the pdf content into PineCone DB

# Demo Video

You could find the demo video in the repo which named "COMS579ProjectFirstDemo.mp4"

# Acknowledgments

The creators of Langchain for their innovative approach to language processing.

Pinecone team for providing an efficient vector database solution.

Gradio for making it easy to create web interfaces for machine learning models.
