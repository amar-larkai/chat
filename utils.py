import os
import markdown
import html2text
import fnmatch
import requests
from bs4 import BeautifulSoup


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from openai.embeddings_utils import get_embedding
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader 
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.llms import AzureOpenAI
import openai

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# csv_path = ""
# csv_data = CSVLoader(file_path=csv_path).load()


# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
# )
# chunks = text_splitter.split_documents(csv_data)

# def get_csv_data():
#     print('gathering csv data')
#     csv_path = ""
#     csv_data = CSVLoader(file_path=csv_path).load()
#     return csv_data

def get_pdf_data():
    print("Gathering ECG Information")
    pdf_path = "ecg_pdf_book.pdf"
    pdf_data = PyPDFLoader(file_path=pdf_path).load()
    return pdf_data





def get_embeddings(input_texts):
    print("Creating Embeddings....")
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(input_texts, embeddings)
    return docsearch

# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     temperature=0,
#     openai_api_key=OPENAI_API_KEY,
#     max_tokens=512,
# )

# def get_model():
#     print("Building OpenAI model....")
#     chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(),
#     chain_type_kwargs={"prompt": SUPPORT_PROMPT},
# )
#     return chain

def get_model():
    print("Building OpenAI model....")
    chain = load_qa_chain(
        ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=512,
        ),
        chain_type="stuff",
    )
    return chain
