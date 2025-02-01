from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
import json
from typing import List, Dict
from datetime import datetime
import re

def extract_metadata(text: str) -> Dict:
    """Extract metadata from text content."""
    metadata = {
        "document_type": "Union Budget Speech",
        "year": "2025-2026",
        "country": "India",
        "speaker": "Nirmala Sitharaman",
        "role": "Minister of Finance",
        "date": "February 1, 2025"
    }
    
    # Extract part information
    part_match = re.search(r'PART [AB]', text)
    if part_match:
        metadata["part"] = part_match.group(0)
    
    # Extract section information
    section_match = re.search(r'^(\d+)\.\s*(.+?)(?=\n|$)', text, re.MULTILINE)
    if section_match:
        metadata["section_number"] = section_match.group(1)
        metadata["section_title"] = section_match.group(2).strip()
    
    return metadata

def create_langchain_chunks(input_file: str, output_file: str):
    """Create chunks using LangChain's advanced text splitting."""
    
    # Load the document
    loader = TextLoader(input_file, encoding='utf-8')
    documents = loader.load()
    
    # Create text splitter with optimal parameters for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    # Process chunks and add metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Extract metadata for this chunk
        chunk_metadata = extract_metadata(chunk.page_content)
        
        # Create chunk dictionary with enhanced structure
        processed_chunk = {
            "chunk_id": f"chunk_{i+1}",
            "content": chunk.page_content,
            "metadata": {
                **chunk_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tokens_estimate": len(chunk.page_content.split()),
                "char_length": len(chunk.page_content),
                "context_window": {
                    "start_index": i * (500 - 50) if i > 0 else 0,
                    "end_index": (i + 1) * 500
                }
            },
            "relationships": {
                "previous_chunk": f"chunk_{i}" if i > 0 else None,
                "next_chunk": f"chunk_{i+2}" if i < len(chunks)-1 else None
            }
        }
        processed_chunks.append(processed_chunk)
    
    # Create final JSON structure
    output_json = {
        "document_metadata": {
            "title": "Union Budget Speech 2025-2026",
            "source": "Ministry of Finance, Government of India",
            "document_type": "Budget Speech",
            "year": "2025-2026",
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "chunk_size": 500,
                "chunk_overlap": 50,
                "total_chunks": len(chunks),
                "splitter_type": "RecursiveCharacterTextSplitter",
                "version": "1.0"
            }
        },
        "chunks": processed_chunks
    }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    return len(processed_chunks)

def main():
    input_file = 'budget_speech.txt'
    output_file = 'budget_speech_chunks_langchain.json'
    
    num_chunks = create_langchain_chunks(input_file, output_file)
    print(f"Successfully created {num_chunks} chunks using LangChain")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
print("Created files:", ["budget_speech_chunks_langchain.json"])