from openai import OpenAI
import json
import os
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_json_chunks(file_path):
    """Load JSON chunks from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['chunks']

def create_embedding(text, retries=3):
    """Create embeddings for a single text using OpenAI API with retry logic."""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                print(f"Failed to create embedding after {retries} attempts: {e}")
                return None
            time.sleep(1)  # Wait before retrying

def process_chunks_with_embeddings(chunks, batch_size=20):
    """Process chunks and add embeddings with rate limiting and batching."""
    chunks_with_embeddings = []
    successful_embeddings = 0
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        
        for chunk in batch:
            embedding = create_embedding(chunk['content'])
            
            if embedding:
                chunk_with_embedding = {
                    **chunk,
                    'embedding': embedding
                }
                chunks_with_embeddings.append(chunk_with_embedding)
                successful_embeddings += 1
            
            # Rate limiting
            time.sleep(0.1)
    
    print(f"Successfully created embeddings for {successful_embeddings} chunks")
    return chunks_with_embeddings

def save_embeddings(chunks_with_embeddings, output_file):
    """Save chunks with embeddings to file."""
    if not chunks_with_embeddings:
        raise ValueError("No embeddings were created successfully")
    
    output_data = {
        "metadata": {
            "model": "text-embedding-ada-002",
            "embedding_dimension": len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else 0,
            "total_chunks": len(chunks_with_embeddings),
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "chunks": chunks_with_embeddings
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Embeddings saved to: {output_file}")

def main():
    # Input and output files
    input_file = 'budget_speech_chunks_langchain.json'
    output_file = 'budget_speech_embeddings.json'
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it using:\nexport OPENAI_API_KEY='your-api-key'")
    
    print("Loading chunks...")
    chunks = load_json_chunks(input_file)
    
    # Process chunks and create embeddings
    chunks_with_embeddings = process_chunks_with_embeddings(chunks)
    
    if chunks_with_embeddings:
        print("Saving embeddings...")
        save_embeddings(chunks_with_embeddings, output_file)
    else:
        print("No embeddings were created successfully")

if __name__ == "__main__":
    try:
        main()
        print("\nCreated files:", ["budget_speech_embeddings.json"])
    except Exception as e:
        print(f"Error: {str(e)}")