raise Exception("DOGRU APP.PY CALISIYOR")

import streamlit as st
from pypdf import PdfReader
import openai
import faiss
import numpy as np

SYSTEM_PROMPT = """
Sen bir İŞ PAKETİ DENETÇİ AJANISIN.
Yalnızca yüklenen PDF içeriğine dayanarak cevap ver.
PDF dışında bilgi uydurma.
"""

import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

st.title("PDF Denetçi Ajanı")

uploaded_file = st.file_uploader("PDF yükle", type="pdf")

@st.cache_resource
def build_index(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        embeddings.append(emb)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

if uploaded_file:
    reader = PdfReader(uploaded_file)
    pages = [p.extract_text() or "" for p in reader.pages]

    chunks = []
    for i, p in enumerate(pages):
        for part in p.split("\n\n"):
            if len(part) > 200:
                chunks.append(f"(Sayfa {i+1}) {part}")

    index = build_index(chunks)

    question = st.chat_input("PDF hakkında sor")

    if question:
        q_emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        D, I = index.search(
            np.array([q_emb]).astype("float32"), k=5
        )
        context = "\n\n".join([chunks[i] for i in I[0]])

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context + "\n\n" + question}
            ]
        )

        st.write(response.choices[0].message.content)
