import os
import streamlit as st
from pypdf import PdfReader
import openai
import faiss
import numpy as np

# =========================
# SABİT TALİMAT (KULLANICI DEĞİŞTİREMEZ)
# =========================
SYSTEM_PROMPT = """
Sen bir İŞ PAKETİ DENETÇİ AJANISIN.
Yalnızca yüklenen PDF içeriğine dayanarak cevap ver.
PDF dışında bilgi uydurma.
Yanıt verirken mümkünse sayfa referansı belirt.
"""

# =========================
# OPENAI KEY (Render Environment Variable)
# =========================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY environment variable bulunamadı. Render > Environment kısmına ekleyin.")
    st.stop()

openai.api_key = api_key

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PDF Denetçi Ajanı", layout="wide")
st.title("PDF Denetçi Ajanı")

uploaded_file = st.file_uploader("PDF yükle", type=["pdf"])

# -------------------------
# Helpers
# -------------------------
def pdf_to_pages_text(pdf_file) -> list[str]:
    reader = PdfReader(pdf_file)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return pages

def make_chunks(pages: list[str]) -> list[str]:
    chunks = []
    for i, page_text in enumerate(pages):
        # Paragraf bazlı bölme; çok kısa parçaları eleme eşiğini düşük tutuyoruz
        parts = [x.strip() for x in page_text.split("\n\n") if x.strip()]
        for part in parts:
            if len(part) >= 60:
                chunks.append(f"(Sayfa {i+1}) {part}")
    return chunks

@st.cache_resource
def build_index(text_chunks: list[str]):
    # PDF’den metin çıkmadıysa
    if not text_chunks:
        return None, None

    embeddings = []
    for chunk in text_chunks:
        emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        embeddings.append(emb)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return index, text_chunks

def retrieve_context(index, chunks: list[str], question: str, k: int = 5) -> str:
    q_emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    D, I = index.search(np.array([q_emb], dtype="float32"), k=min(k, len(chunks)))
    picked = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return "\n\n".join(picked)

def ask_llm(context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PDF BAĞLAMI:\n{context}\n\nSORU:\n{question}\n\nKURALLAR:\n- Yalnızca PDF bağlamına dayan.\n- Bilgi yoksa 'PDF içinde bu bilgi yok' de.\n- Uydurma yapma."}
    ]
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return resp.choices[0].message.content

# -------------------------
# Main flow
# -------------------------
if uploaded_file is None:
    st.info("Başlamak için bir PDF yükleyin.")
    st.stop()

# PDF işle
with st.spinner("PDF okunuyor ve indeksleniyor..."):
    pages = pdf_to_pages_text(uploaded_file)
    chunks = make_chunks(pages)
    index, chunks = build_index(chunks)

if index is None:
    st.error("PDF içinden okunabilir metin çıkarılamadı. Bu PDF tarama (görüntü) olabilir.")
    st.stop()

st.success("PDF hazır. Şimdi soru sorabilirsiniz.")

# Chat geçmişi
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("PDF hakkında sorunuz...")

if question:
    with st.spinner("Cevap hazırlanıyor..."):
        context = retrieve_context(index, chunks, question, k=5)
        answer = ask_llm(context, question)

    st.session_state.history.append((question, answer))

# Geçmişi göster
for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

