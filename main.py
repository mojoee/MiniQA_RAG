from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv
from utils import load_data, preprocess_pdfs, is_collection_loaded
import json
import os
import socket
from pymilvus import Collection, FieldSchema, CollectionSchema, connections, Index, IndexType, DataType
from openai import OpenAI


load_dotenv()  # Load environment variables from .env file

CREATE_COLLECTION=False
DELETE_COLLECTION=False  # Set this to True if you want to delete the collection
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
COLLECTION_NAME = 'test_collection'
PORT = 12345
SERVER_IP='0.0.0.0'
GENERATIVE_MODEL_NAME = "gpt-3.5-turbo"


if __name__ =="__main__":
    logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum logging level you want to log
    format='%(asctime)s - %(levelname)s - %(message)s'
    # You can customize the format of your log messages here
    )

    # load model
    model = SentenceTransformer(MODEL_NAME)
    logging.info(f"Model: {MODEL_NAME}")

    # Extract text from your PDF files
    if CREATE_COLLECTION:
        data = load_data()
        preprocessed_texts = preprocess_pdfs(data)
        embeddings = model.encode(preprocessed_texts)
        logging.debug("data loaded and text preprocessed")


    # Initialize OpenSearch client with authentication
    # Connect to Milvus server
    connections.connect(host='localhost', port='19530')

    # Define collection schema 
    schema = CollectionSchema(fields=[
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),  # Primary key field
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1200),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=len(model.encode("Sample text.")))  # Adjust dim to match your embedding dimensions
    ])

    # Create collection
    
    # delete old collection
    if Collection(COLLECTION_NAME).schema:
        if DELETE_COLLECTION:
            Collection(COLLECTION_NAME).drop()
            print(f"Collection '{COLLECTION_NAME}' dropped successfully.")
    
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    

    # Insert embeddings into the collection
    if CREATE_COLLECTION:
        ids = [i for i in range(len(embeddings))] 
        sources = [str(doc) for doc in preprocessed_texts]
        data = [{'id': id, 'source': doc, 'embedding': embedding.tolist()} for id, doc, embedding in zip(ids, sources, embeddings)]
        ids = collection.insert(data)

    # Define index parameters
    index_param = {
        'metric_type': 'L2',  # Specify the distance metric (L2 for Euclidean distance)
        'index_type': IndexType.FLAT,  # Specify the index type (e.g., IVF_FLAT)
        'params': {'nlist': 100}  # Adjust nlist based on your data size and query requirements
    }

    # Create index on the collection
    index = Index(collection, field_name='embedding', index_params=index_param)
    collection.create_index('id', index)

    # socket
    # Define server address and port
    server_address = (SERVER_IP, PORT)

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)
    print(f"Server is listening on {server_address}")

    while True:
        # Wait for a connection
        connection, client_address = server_socket.accept()
        try:
            logging.info(f"Connection from {client_address}")
            data = connection.recv(512)
            logging.info(f"Received from server: {data.decode()}")
            query = data.decode()
            # Define query vector (assuming a 512-dimensional query vector)
            # query = "Tell me about Priority for client orders: order handling and recording"
            # query = "Tell me about the use of special purpose vehicles."
            query_embedding = model.encode(query)
            search_params = {
                'metric_type': 'L2',  # Specify the distance metric (L2 for Euclidean distance)
                'params': {'nprobe': 10}  # Adjust nprobe based on your data size and query requirements
            }
            # Usage:
            if not is_collection_loaded(collection):
                # Collection is loaded, proceed with search operation
                print("Collection is not loaded. Attempting to load...")
                collection.load()

            # Now try to perform the search operation again
            results = collection.search([query_embedding], 
                                    top_k=10, 
                                    limit=3, 
                                    output_fields=['source'],
                                    anns_field='embedding',  # Assuming 'embedding' is the name of your embedding field
                                    param=search_params,)  # Adjust top_k based on your retrieval needs
            # Extract and format the search results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_result = {
                        'id': hit.id,  # Assuming 'id' is the primary key or identifier associated with the hit
                        'distance': hit.distance,  # Distance from the query vector
                        'source': hit.entity.get('source'),  # Assuming 'source' is a field in your Milvus collection
                    }
                formatted_results.append(formatted_result)

            # Convert the results to a JSON string
            json_results = json.dumps(formatted_results)
            logging.info(f"Sending results to client: {json_results}")

            # Construct a prompt based on search results
            prompt = "Please answer the following query: "
            prompt += query
            prompt += "Given the following context."
            for i, result in enumerate(formatted_results, start=1):
                prompt += f"{i}. Source: {result['source']}\n"
                prompt += f"   Distance: {result['distance']}\n\n"

            logging.info(f"Constructed prompt:\n{prompt}")
            client = OpenAI()
            message = {
                'role': 'user',
                'content': prompt
            }

            # Send request to OpenAI API
            response = client.chat.completions.create(
                model=GENERATIVE_MODEL_NAME,
                messages=[message]
            )
            chatbot_response = response.choices[0].message.content

            # Send the JSON results to the client (if needed)
            connection.sendall(chatbot_response.encode('utf-8'))
        finally:
            # Clean up the connection
            connection.close()



