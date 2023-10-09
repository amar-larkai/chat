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

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import numpy as np
import torch
import cv2
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

config=ViTConfig()

config.hidden_size=1024
config.intermediate_size=4096
config.num_attention_heads=16
config.num_hidden_layers=24
config.patch_size=16

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


class ImageDataset(torch.utils.data.Dataset):

  def __init__(self, input_data):

      self.input_data = input_data
      # Transform input data
      self.transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        ])

  def __len__(self):
      return len(self.input_data)

  def get_images(self, idx):
      return self.transform(self.input_data[idx]['image'])

  def get_labels(self, idx):
      return self.input_data[idx]['label']

  def __getitem__(self, idx):
      # Get input data in a batch
      train_images = self.get_images(idx)
      train_labels = self.get_labels(idx)

      return train_images, train_labels
  

label2id = {'myocardial infarctions': 0, 'normal': 1}
id2label = {0: 'myocardial infarctions', 1: 'normal'}

class ViT(nn.Module):

  def __init__(self, config=config, num_labels=2,
               model_checkpoint='google/vit-large-patch16-224-in21k'):

        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels)
        )

  def forward(self, x):

    x = self.vit(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output
  

def load_model():
    with open('model.pkl', 'rb') as f:
        model_state_dict = pickle.load(f)  
    model = ViT()
    model.load_state_dict(model_state_dict)  
    return model  