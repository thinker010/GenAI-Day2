import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import os

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 

st.title("📄 PDF Analyzer with Gemini Chatbot")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Load the PDF content
    with st.spinner("Loading PDF..."):
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        pdf_text = "\n\n".join(page.page_content for page in documents)  # Combine all pages

    st.success("PDF loaded successfully!")

    # User Query
    query = st.text_input("Enter your query for this PDF (e.g., Summarize, Extract key points, etc.)")

    if query:
        prompt = PromptTemplate.from_template(query + " {text}")
        chain = prompt | llm  # Create the chain

        with st.spinner("Analyzing PDF..."):
            result = chain.invoke({"text": pdf_text})

        st.subheader("📌 Result:")
        st.write(result.content)
