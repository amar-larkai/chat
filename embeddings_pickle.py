import os
import random
import time
import pickle

from utils import  get_embeddings, get_model, get_pdf_data

from langchain.text_splitter import RecursiveCharacterTextSplitter




pdf_data=get_pdf_data()
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(pdf_data)
embeddings = get_embeddings(chunks)
#chain = get_model()
with open("embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)