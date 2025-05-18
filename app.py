import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import requests
import os

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 

st.title("🌐 PDF Analyzer with Gemini Chatbot (Direct GitHub PDF)")

# Direct GitHub raw URL of the PDF
github_pdf_url = "https://raw.githubusercontent.com/thinker010/GenAI-Day2/main/2505.06633v1.pdf"

try:
    with st.spinner("Downloading PDF from GitHub..."):
        response = requests.get(github_pdf_url)
        if response.status_code == 200:
            pdf_path = "temp_github_pdf.pdf"
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            
            with st.spinner("Loading PDF..."):
                loader = PyPDFLoader(pdf_path)
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

            # Clean up the downloaded PDF
            os.remove(pdf_path)
        else:
            st.error(f"Failed to download PDF. HTTP Status Code: {response.status_code}")

except Exception as e:
    st.error(f"Failed to process the PDF: {str(e)}")
