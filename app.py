import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# Must be the first Streamlit command
st.set_page_config(page_title="Telugu LLM", layout="centered")

# Debug output: confirm current folder and files
st.write("📁 Current working directory:", os.getcwd())
st.write("📂 Files in folder:", os.listdir())

# ✅ Correct Excel file name
excel_path = os.path.join(os.getcwd(), "local_history_data.xlsx")

if not os.path.exists(excel_path):
    st.error("❌ 'local_history_data.xlsx' not found in:\n" + excel_path)
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_excel(excel_path)
    return df.dropna(subset=["place", "description"])

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load data and model
df = load_data()
model = load_model()

# Encode descriptions once
corpus = df["description"].tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Simple user input
query = st.text_input("ప్రశ్నను తెలుగు లో టైప్ చేయండి:")

# Matching and output
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    threshold = 0.5
    matched_idxs = [i for i, score in enumerate(scores) if score >= threshold]

    if matched_idxs:
        matched_texts = [df.iloc[i]["description"] for i in matched_idxs]

        # ✅ Remove duplicate sentences
        all_sentences = []
        seen = set()
        for desc in matched_texts:
            for sentence in desc.replace("।", ".").split("."):
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    all_sentences.append(sentence)

        final_paragraph = ". ".join(all_sentences) + "."
        st.markdown(final_paragraph)
    else:
        st.markdown("❌ సంబంధిత సమాచారం కనిపించలేదు.")
