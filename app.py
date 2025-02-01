import streamlit as st
from openai import OpenAI
import json
import numpy as np
from numpy.linalg import norm
import os
from typing import List, Dict
import re

def init_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            st.error("OpenAI API key not found in secrets.")
            st.stop()
        return OpenAI(
            api_key=api_key,
            timeout=60.0  # Adding a timeout
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.stop()

client = init_openai_client()

if not client:
    st.error("Failed to initialize OpenAI client")
    st.stop()

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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return original_query

#RAG funcitons will start here
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
        response = client.embeddings.create(
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"}
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=900
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit user interface
st.set_page_config(layout="wide", page_title="Budget 2025-26 Assistant")

# Sidebar
with st.sidebar:
    st.title("📚 Guide")
    st.markdown("""
    ### How to Use
    1. Ask any question about Budget 2025-26
    2. For calculations, simply ask naturally
    3. View sources for transparency
    4. Inquire about specific schemes or initiatives

    ### Important Note
    This assistant provides answers based exclusively on the Union Budget 2025-26 data presented by the Finance Minister on February 1, 2025. Information outside this scope or about subsequent modifications may not be available.

    ### Sample Questions
    - What are the key highlights of Budget 2025-26?
    - Calculate tax for income of 45 lakhs under new regime
    - What are the major infrastructure projects announced?
    - Show the breakdown of healthcare spending
    - What are the changes in income tax slabs?
    - What is the fiscal deficit target?
    - Compare agriculture budget with previous year
    - What are the new schemes announced for startups?
    """)

# Main content
st.title("🏛️ Budget 2025-26 Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'cached_responses' not in st.session_state:
    st.session_state.cached_responses = {}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the Budget..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Processing...")

        try:
            # Check cache first
            if prompt in st.session_state.cached_responses:
                response = st.session_state.cached_responses[prompt]
                relevant_chunks = st.session_state.cached_responses[f"{prompt}_chunks"]
            else:
                chunks = load_embeddings()
                relevant_chunks = semantic_search(prompt, chunks)

                if not relevant_chunks:
                    message_placeholder.markdown("❌ No relevant information found.")
                    st.stop()

                context = "\n".join([chunk['content'] for chunk in relevant_chunks])
                response = get_chat_response(prompt, context)

                # Cache the response and chunks
                st.session_state.cached_responses[prompt] = response
                st.session_state.cached_responses[f"{prompt}_chunks"] = relevant_chunks

            message_placeholder.markdown(response)

            with st.expander("📑 Sources"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Source {i}:**\n{chunk['content']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            message_placeholder.markdown(f"❌ Error: {str(e)}")
