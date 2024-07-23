def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    index = 0
    
    while index < len(text):
        if index + chunk_size < len(text):
            chunks.append(text[index:index + chunk_size])
        else:
            chunks.append(text[index:])
        index += chunk_size - overlap  # Move forward with overlap
    
    return chunks

# Example usage
def process_documents(documents):
    chunked_texts = []
    for document in documents:
        chunks = chunk_text(document, chunk_size=500)
        chunked_texts.extend(chunks)
    
    return chunked_texts

# Example of how to use the function
if __name__ == "__main__":
    # Example documents (replace this with your actual loaded PDF texts)
    documents = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    ]
    
    # Process documents into chunks
    chunked_texts = process_documents(documents)
    
    # Print each chunked text
    for i, chunk in enumerate(chunked_texts):
        print(f"Chunk {i + 1}: {chunk}")
