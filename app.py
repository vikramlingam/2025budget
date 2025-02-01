import streamlit as st
import openai  # Correct import for the OpenAI library
import json
import numpy as np
from numpy.linalg import norm
import os
from typing import List, Dict
import re

# Initialize OpenAI client with proper error handling
def init_openai_client():
    """Initialize OpenAI client with proper error handling"""
    try:
        if not st.secrets.get("OPENAI_API_KEY"):
            st.error("OpenAI API key not found in secrets!")
            st.stop()

        # Set the API key for the openai library
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        st.stop()

# Initialize the client
init_openai_client()

def convert_indian_format(value_str: str) -> float:
    """Convert Indian number format (crores, lakhs) to float"""
    value_str = value_str.lower().replace(',', '')
    multiplier = 1

    if 'crore' in value_str:
        multiplier = 10000000
        value_str = value_str.replace('crore', '').replace('crores', '')
    elif 'lakh' in value_str:
        multiplier = 100000
        value_str = value_str.replace('lakh', '').replace('lakhs', '')

    value_str = value_str.replace('₹', '').replace('rs', '').replace('rs.', '').strip()
    try:
        return float(value_str) * multiplier
    except ValueError:
        return 0.0

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numerical values with Indian format support"""
    pattern = r'(?:Rs\.?|₹)?(?:\s*)?(\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh))?)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [convert_indian_format(match) for match in matches if match]

def format_indian_currency(amount: float) -> str:
    """Format amount in Indian currency format with crores and lakhs"""
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Crore"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} Lakh"
    else:
        s = f"{amount:,.2f}"
        parts = s.split('.')
        parts[0] = re.sub(r'(\d)(?=(\d\d\d)+(?!\d))', r'\1,', parts[0])
        return f"₹{parts[0]}"

@st.cache_data(ttl=3600)
def optimize_prompt(original_query: str) -> str:
    """Optimize user query for better retrieval and calculation accuracy"""
    try:
        messages = [
            {
                "role": "system",
                "content": "Rephrase the query for better retrieval. Keep it concise."
            },
            {
                "role": "user",
                "content": original_query
            }
        ]

        response = openai.ChatCompletion.create(  # Use openai module directly
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return original_query

# RAG functions will start here
@st.cache_data
def load_embeddings() -> List[Dict]:
    """Load pre-computed embeddings"""
    try:
        with open('budget_speech_embeddings.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['chunks']
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return []

@st.cache_data(ttl=3600)
def create_embedding(text: str) -> List[float]:
    """Create embedding for search query with caching"""
    try:
        response = openai.Embedding.create(  # Use openai module directly
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return None

@st.cache_data(ttl=3600)
def semantic_search(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Optimized semantic search with caching"""
    query_embedding = create_embedding(query)
    if not query_embedding:
        return []

    # Using numpy for faster computation, otherwise there is a delay
    chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
    query_embedding = np.array(query_embedding)

    # Vectorized similarity computation
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [chunks[i] for i in top_indices]

def analyze_calculation_request(query: str) -> dict:
    """Streamlined calculation analysis"""
    # Quick check without API call
    calculation_keywords = ['calculate', 'compute', 'what is the tax', 'percentage',
                          'growth rate', 'difference between']
    if not any(keyword in query.lower() for keyword in calculation_keywords):
        return {"needs_calculation": False}

    try:
        messages = [
            {
                "role": "system",
                "content": "Analyze if calculation needed. Return JSON with needs_calculation (boolean) and calculation_type (string)."
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }
        ]

        response = openai.ChatCompletion.create(  # Use openai module directly
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=400
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"needs_calculation": False}

def get_chat_response(query: str, context: str) -> str:
    """Optimized response generation"""
    try:
        # Skip optimization for simple queries to make it process faster
        if len(query.split()) < 5 and not any(word in query.lower() for word in ['calculate', 'tax', 'percentage']):
            optimized_query = query
        else:
            optimized_query = optimize_prompt(query)

        analysis = analyze_calculation_request(optimized_query)

        messages = [
            {
                "role": "system",
                "content": """Provide accurate, concise responses. For calculations, show clear steps.
                Use Indian currency format. Focus on directly relevant information."""
            },
            {
                "role": "user",
                "content": f"Query: {optimized_query}\nContext: {context}\nNeeds Calculation: {analysis.get('needs_calculation', False)}"
            }
        ]

        response = openai.ChatCompletion.create(  # Use openai module directly
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=900
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
