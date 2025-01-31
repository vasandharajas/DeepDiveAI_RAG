import chardet
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Set your Hugging Face API token
hf_token = 'your api key'
os.environ[''] = '1'  # Suppress symlink warning

# Load Hugging Face models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=hf_token)
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased", token=hf_token)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load your pre-trained SentenceTransformer model for semantic search
semantic_model = SentenceTransformer('pre-trained model')

# Streamlit app
st.title("Research Paper Query System")
st.write("Upload research papers and input queries to get AI-generated responses.")

# Function to perform semantic search
def retrieve_chunks_from_vector_db(query_embedding, database_embeddings, top_k=5):
    similarities = cosine_similarity(query_embedding, database_embeddings)
    top_k_indices = np.argsort(similarities[0])[::-1][:top_k]
    return [research_chunks[i] for i in top_k_indices]

# Function to extract abstract and conclusion
def extract_sections(text, sections=('abstract', 'conclusion')):
    abstract = ""
    conclusion = ""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.lower().startswith('abstract'):
            abstract = " ".join(lines[i:])
        elif line.lower().startswith('conclusion'):
            conclusion = " ".join(lines[i:])
    return abstract, conclusion

# Function to process a single query
def process_single_query(query, database_chunks, database_embeddings):
    query_embedding = semantic_model.encode([query])
    relevant_chunks = retrieve_chunks_from_vector_db(query_embedding, database_embeddings)
    context = " ".join(relevant_chunks)
    
    # Pass the context and query to the Hugging Face model for generating a response
    response = qa_pipeline(question=query, context=context)
    return response['answer']

# Upload research papers
uploaded_files = st.file_uploader("Upload Research Papers", type="txt", accept_multiple_files=True)

# Store research chunks
research_chunks = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        text = raw_data.decode(encoding)
        abstract, conclusion = extract_sections(text)
        research_chunks.append(abstract)
        research_chunks.append(conclusion)
    
    # Encode the research chunks
    database_embeddings = semantic_model.encode(research_chunks)

# Input query
query = st.text_input("Input your query")

if st.button("Get Response"):
    if research_chunks and query:
        response = process_single_query(query, research_chunks, database_embeddings)
        st.write(f"Query: {query}")
        st.write(f"Response: {response}")
    else:
        st.write("Please upload research papers and input a query.")
