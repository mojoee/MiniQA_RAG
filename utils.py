import PyPDF2
from nltk.tokenize import sent_tokenize
import os
import nltk
from dotenv import load_dotenv
from pymilvus import MilvusClient
from text_processing import process_documents
import pymilvus


load_dotenv()


def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text


def preprocess_text(text):
    # Tokenize text into sentences
    # sentences = sent_tokenize(text)
    # can add more cleaning here if necessary
    chunked_texts = process_documents(text)
    chunked_texts = [str(sent_tokenize(text)) for text in chunked_texts]
    return chunked_texts


def load_data(data_dir='data'):
    pdf_texts = []
    source_files = os.listdir(data_dir)
    for pdf_file in source_files:
        pdf_texts.append(extract_text_from_pdf(f"data/{pdf_file}"))
    return pdf_texts


def preprocess_pdfs(pdf_texts):
    # Download necessary NLTK data
    nltk.download('punkt')
    # Preprocess each extracted PDF text
    preprocessed_texts = preprocess_text(pdf_texts)
    return preprocessed_texts


def get_client():
    # Load environment variables from .env file
    # Retrieve username and password from environment variables
    client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="default"
    )
    return client


def check_client(client, logging):
    # Check if the cluster is up and healthy
    response = client.ping()
    if response:
        health = client.cluster.health()
        logging.info("Health")
        logging.info("Cluster is up and healthy!")
    else:
        logging.info("Cluster is down or unreachable!")


def is_collection_loaded(collection):
    try:
        # Attempt to perform a simple operation that requires the collection to be loaded
        if collection.num_entities > 0:
            return True
    except pymilvus.exceptions.MilvusException:
        return False