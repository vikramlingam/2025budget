https://2025budget.streamlit.app

# Budget 2025-26 Assistant - Retrieval-Augmented Generation (RAG) App

## Introduction
The **Budget 2025-26 Assistant** is a Streamlit-based **Retrieval-Augmented Generation (RAG)** application designed to answer queries related to the **Indian Union Budget 2025-26**. This app leverages OpenAI's GPT-3.5-turbo model to provide accurate and context-aware answers. The budget document (PDF) was pre-processed and converted into embeddings, enabling the app to retrieve relevant information efficiently and generate responses based on user queries.

This tool is ideal for individuals, researchers, and professionals seeking quick insights into the Union Budget 2025-26, including tax calculations, policy highlights, and financial data.

---

## Features
- **Query Answering**: Ask questions about the Union Budget 2025-26, and the app retrieves relevant information and generates concise answers.
- **Retrieval-Augmented Generation (RAG)**: Combines semantic search with GPT-3.5-turbo to provide accurate and contextually relevant responses.
- **Indian Number Format Support**: Handles Indian currency formats (crores, lakhs) for calculations and displays.
- **Semantic Search**: Uses embeddings to retrieve the most relevant sections of the budget document for each query.
- **Rate Limit Handling**: Automatically retries API calls when OpenAI rate limits are reached.
- **Caching**: Speeds up performance by caching embeddings, optimized prompts, and responses.

---

## How It Works
1. **Embedding Creation**: The budget PDF was pre-processed, and its content was divided into chunks. Each chunk was converted into embeddings using OpenAI's `text-embedding-ada-002` model.
2. **Semantic Search**: When a user submits a query, the app creates an embedding for the query and performs a similarity search against the pre-computed embeddings to retrieve the most relevant chunks.
3. **Answer Generation**: The retrieved chunks are passed as context to OpenAI's GPT-3.5-turbo model, which generates a detailed and accurate response.
4. **Rate Limit Handling**: The app includes mechanisms to handle OpenAI's rate limits by retrying requests after a delay.
5. **User Interface**: A simple and intuitive Streamlit interface allows users to interact with the app, view responses, and explore sources.

---

## Installation
Follow these steps to set up and run the app locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/budget-2025-26-assistant.git
   cd budget-2025-26-assistant
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up OpenAI API Key**:
   - Create a `secrets.toml` file in the `.streamlit` directory:
     ```bash
     mkdir -p .streamlit
     echo "[secrets]" > .streamlit/secrets.toml
     echo "OPENAI_API_KEY = 'your-openai-api-key'" >> .streamlit/secrets.toml
     ```
   - Replace `'your-openai-api-key'` with your actual OpenAI API key.

4. **Run the App**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

---

## Usage
1. **Ask Questions**:
   - Enter your query in the input box (e.g., "What are the key highlights of Budget 2025-26?" or "Calculate tax for income of 45 lakhs under the new regime").
2. **View Responses**:
   - The app retrieves relevant information from the budget document and generates a detailed response.
3. **Explore Sources**:
   - Expand the "Sources" section to view the specific sections of the budget document used to generate the response.

### Sample Queries
- "What are the major infrastructure projects announced?"
- "What is the fiscal deficit target for 2025-26?"
- "Compare the agriculture budget with the previous year."
- "What are the changes in income tax slabs?"

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **OpenAI**: For providing the GPT-3.5-turbo and embedding models.
- **Streamlit**: For the interactive web app framework.
- **Government of India**: For the Union Budget 2025-26 document.

---

Feel free to contribute to this project by submitting issues or pull requests. For any questions, contact [your-email@example.com].
