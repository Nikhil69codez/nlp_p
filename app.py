import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx2txt
from model_logic import train_model

# Helper function to extract text based on file type
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return None

st.title("📂 Smart Multi-Format Document Classifier")
tfidf, model = train_model()

uploaded_file = st.file_uploader("Upload a Document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if uploaded_file:
    document_text = extract_text(uploaded_file)
    st.info(f"Loaded {uploaded_file.name}")

    if st.button("Classify Document"):
        vec = tfidf.transform([document_text])
        res = model.predict(vec)[0]
        st.success(f"Classification: {res.upper()}")
        
        # Visualization
        probs = model.predict_proba(vec)[0]
        st.bar_chart(pd.DataFrame(probs, index=model.classes_, columns=["Confidence"]))