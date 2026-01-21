import os
import streamlit as st
import openai
import faiss
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO
import streamlit as st

st.set_page_config(
    page_title="İş Paketi Denetçi Ajanı",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Üst bant
st.markdown("""
<style>
/* Genel tipografi ve boşluklar */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
.small-muted { color: #6b7280; font-size: 0.9rem; }

/* Header bandı */
.header {
  padding: 12px 16px;
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 10px;
  background: rgba(240,240,240,.35);
}
.kpi {
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 10px;
  padding: 10px 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">', unsafe_allow_html=True)
colA, colB = st.columns([1, 3], vertical_alignment="center")
with colA:
    # Logo dosyanız varsa repo içine koyup buradan gösterin (örn: assets/logo.png)
    # st.image("assets/logo.png", width=90)
    st.write("LOGO")
with colB:
    st.markdown("## İş Paketi Denetçi Ajanı")
    st.markdown('<div class="small-muted">Offline Demo / Kurum içi kurallar sabit • Yalnızca yüklenen içerik temel alınır</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

SYSTEM_PROMPT = """
Sen bir İŞ PAKETİ DENETÇİ AJANISIN.
Yalnızca yüklenen PDF içeriğine dayanarak cevap ver.
PDF dışında bilgi uydurma.
Bilgi yoksa açıkça belirt.
"""

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY Render > Environment Variables kısmında tanımlı değil.")
    st.stop()
openai.api_key = api_key

st.set_page_config(page_title="PDF Denetçi Ajanı", layout="wide")
st.title("PDF Denetçi Ajanı")

uploaded_file = st.file_uploader("PDF yükle", type=["pdf"])

def extract_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for page in doc:
        out.append(page.get_text("text") or "")
    return out

def extract_pages_pdfplumber(pdf_bytes: bytes) -> list[str]:
    out = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return out

def make_chunks(pages: list[str]) -> list[str]:
    chunks = []
    for i, text in enumerate(pages):
        # Daha agresif: çok kısa parçaları da al (metin az çıkıyorsa kaçırmayalım)
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in parts:
            if len(part) >= 20:
                chunks.append(f"(Sayfa {i+1}) {part}")
    return chunks

@st.cache_resource
def build_index(chunks: list[str]):
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

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PDF BAĞLAMI:\n{context}\n\nSORU:\n{question}\n\nKURAL: Yalnızca bu PDF bağlamına dayan."}
        ]
    )
    return resp.choices[0].message.content

if not uploaded_file:
    st.info("Başlamak için PDF yükleyin.")
    st.stop()

# PDF bytes: stream sorunlarını bitirmek için tek seferde al
pdf_bytes = uploaded_file.getvalue()

with st.spinner("PDF metni çıkarılıyor..."):
    pages = extract_pages_pymupdf(pdf_bytes)
    total_chars = sum(len(x) for x in pages)

    method_used = "PyMuPDF"
    if total_chars < 200:  # çok az çıktıysa fallback dene
        pages2 = extract_pages_pdfplumber(pdf_bytes)
        total_chars2 = sum(len(x) for x in pages2)
        if total_chars2 > total_chars:
            pages = pages2
            total_chars = total_chars2
            method_used = "pdfplumber"

st.caption(f"Metin çıkarma yöntemi: {method_used} | Toplam karakter: {total_chars}")

# Debug: kullanıcıya kanıt göster (ilk 300 karakter)
preview_text = "\n".join(pages).strip()
if preview_text:
    with st.expander("Çıkan metin önizleme (ilk 300 karakter)"):
        st.code(preview_text[:300])
else:
    st.error("PDF içinden metin çıkarılamadı. Bu dosyada metin katmanı yok (image-only) veya çıkarılamayan özel encoding var. OCR gerekir.")
    st.stop()

chunks = make_chunks(pages)

with st.spinner("İndeks hazırlanıyor..."):
    index = build_index(chunks)

if index is None:
    st.error("Metin çıkarıldı ama parçalama sonrası kullanılabilir içerik kalmadı. (Bu PDF çok kısa/dağınık olabilir.)")
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
