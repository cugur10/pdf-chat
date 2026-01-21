import os
import streamlit as st
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np

# =========================
# SABİT TALİMAT
# =========================
SYSTEM_PROMPT = """
Sen bir İŞ PAKETİ DENETÇİ AJANISIN.
Yalnızca yüklenen PDF içeriğine dayanarak cevap ver.
PDF dışında bilgi uydurma.
Bilgi yoksa açıkça belirt.
"""

# =========================
# OPENAI KEY
# =========================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY Render Environment Variables kısmında tanımlı değil.")
    st.stop()

openai.api_key = api_key

# =========================
# UI
# =========================
st.set_page_config(page_title="PDF Denetçi Ajanı", layout="wide")
st.title("PDF Denetçi Ajanı")

uploaded_file = st.file_uploader("PDF yükle", type=["pdf"])

# =========================
# FONKSİYONLAR
# =========================
def extract_pages(pdf_file):
    # Streamlit uploaded_file bir BytesIO gibi davranır
    data = pdf_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text("text") or "")
    return texts


def make_chunks(pages):
    chunks = []
    for i, text in enumerate(pages):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in parts:
            if len(part) >= 60:
                chunks.append(f"(Sayfa {i+1}) {part}")
    return chunks

@st.cache_resource
def build_index(chunks):
    if not chunks:
        return None

    vectors = []
    for c in chunks:
        emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=c
        ).data[0].embedding
        vectors.append(emb)

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))
    return index

def answer_question(index, chunks, question):
    q_emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    D, I = index.search(np.array([q_emb], dtype="float32"), k=min(5, len(chunks)))
    context = "\n\n".join([chunks[i] for i in I[0]])

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PDF:\n{context}\n\nSORU:\n{question}"}
        ]
    )
    return response.choices[0].message.content

# =========================
# AKIŞ
# =========================
if not uploaded_file:
    st.info("Başlamak için PDF yükleyin.")
    st.stop()

with st.spinner("PDF işleniyor..."):
    pages = extract_pages(uploaded_file)
    chunks = make_chunks(pages)
    index = build_index(chunks)

if index is None:
    st.error("PDF içinden okunabilir metin çıkarılamadı (tarama PDF olabilir).")
    st.stop()

st.success("PDF hazır. Sorunuzu yazabilirsiniz.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("PDF hakkında soru sor")

if question:
    with st.spinner("Cevap hazırlanıyor..."):
        answer = answer_question(index, chunks, question)
    st.session_state.history.append((question, answer))

for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

