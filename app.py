import os
from io import BytesIO
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import faiss
import openai

# =========================
# PAGE CONFIG (ONLY ONCE)
# =========================
st.set_page_config(
    page_title="İş Paketi Denetçi Ajanı",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
.small-muted { color: #6b7280; font-size: 0.92rem; }
.header {
  padding: 14px 16px;
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  background: rgba(240,240,240,.35);
}
.kpi {
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,.6);
}
.card {
  border: 1px solid rgba(120,120,120,.25);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(240,240,240,.20);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (SINGLE TITLE)
# =========================
st.markdown('<div class="header">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="center")
with c1:
    # st.image("assets/logo.png", width=90)  # optional
    st.markdown("**LOGO**")
with c2:
    st.markdown("## İş Paketi Denetçi Ajanı")
    st.markdown(
        '<div class="small-muted">Demo • Kurum içi kurallar sabit • Yalnızca yüklenen içerik temel alınır</div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown('<div class="small-muted" style="text-align:right;">Sürüm: v1.0</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================
# SYSTEM PROMPT (FIXED)
# =========================
SYSTEM_PROMPT = """
Sen bir İŞ PAKETİ DENETÇİ AJANISIN.
Yalnızca yüklenen PDF içeriğine dayanarak cevap ver.
PDF dışında bilgi uydurma.
Bilgi yoksa açıkça belirt.
Mümkünse yanıtında sayfa referansı belirt.
"""

# =========================
# API KEY (STREAMLIT SECRETS + ENV FALLBACK)
# =========================
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY bulunamadı. Streamlit Cloud > Secrets veya Environment içine ekleyin.")
    st.stop()

openai.api_key = api_key

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### Menü")
    st.write("• Belge Yükle\n• PDF Metin Durumu\n• Soru-Cevap")
    st.divider()
    st.markdown("### Politika")
    st.caption("Uygulama yalnızca yüklenen PDF içeriğine dayanır. Harici bilgi üretmez.")
    st.markdown("### Gizlilik")
    st.caption("Demo ortamında içerikler kalıcı saklanmamalıdır. Canlı ortamda kurum politikası uygulanır.")

# =========================
# KPI ROW
# =========================
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown('<div class="kpi"><b>1) Belge Yükle</b><br><span class="small-muted">Rapor/İP PDF dosyasını ekleyin.</span></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><b>2) İçerik Doğrulama</b><br><span class="small-muted">Metin çıkarma ve kapsama kontrol edilir.</span></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><b>3) Denetçi Yanıtı</b><br><span class="small-muted">Sorular PDF bağlamından cevaplanır.</span></div>', unsafe_allow_html=True)

st.write("")

# =========================
# PDF UPLOAD
# =========================
uploaded_file = st.file_uploader("Rapor/PDF yükleyin (İP denetimi)", type=["pdf"])

# =========================
# HELPERS
# =========================
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
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        for part in parts:
            if len(part) >= 20:
                chunks.append(f"(Sayfa {i+1}) {part}")
    return chunks

def safe_embedding(text: str) -> list[float]:
    # Quota/Rate limit gibi hatalarda kullanıcıya düzgün mesaj
    try:
        return openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "RateLimitError" in msg or "429" in msg:
            st.error("OpenAI API kota/billing hatası (429). API anahtarınızda kredi/billing yok veya limit dolmuş.")
        else:
            st.error(f"OpenAI API hatası: {e}")
        st.stop()

@st.cache_resource
def build_index(chunks: list[str]):
    if not chunks:
        return None

    vectors = []
    for c in chunks:
        vectors.append(safe_embedding(c))

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))
    return index

def answer_question(index, chunks, question: str) -> str:
    q_emb = safe_embedding(question)

    D, I = index.search(np.array([q_emb], dtype="float32"), k=min(5, len(chunks)))
    context = "\n\n".join([chunks[i] for i in I[0] if 0 <= i < len(chunks)])

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"PDF BAĞLAMI:\n{context}\n\nSORU:\n{question}\n\nKURAL: Yalnızca bu PDF bağlamına dayan."}
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "RateLimitError" in msg or "429" in msg:
            return "OpenAI API kota/billing hatası (429). API anahtarınızda kredi/billing yok veya limit dolmuş."
        return f"OpenAI API hatası: {e}"

# =========================
# FLOW
# =========================
if not uploaded_file:
    st.markdown(
        '<div class="card"><b>Başlamak için</b><br>'
        '<span class="small-muted">Lütfen bir PDF yükleyin. Yüklendikten sonra metin çıkarma ve soru-cevap aktif olacaktır.</span></div>',
        unsafe_allow_html=True
    )
    st.stop()

pdf_bytes = uploaded_file.getvalue()

with st.spinner("PDF metni çıkarılıyor..."):
    pages = extract_pages_pymupdf(pdf_bytes)
    total_chars = sum(len(x) for x in pages)
    method_used = "PyMuPDF"

    if total_chars < 200:
        pages2 = extract_pages_pdfplumber(pdf_bytes)
        total_chars2 = sum(len(x) for x in pages2)
        if total_chars2 > total_chars:
            pages = pages2
            total_chars = total_chars2
            method_used = "pdfplumber"

st.markdown(
    f'<div class="card"><b>PDF Durumu</b><br>'
    f'<span class="small-muted">Metin çıkarma yöntemi: <b>{method_used}</b> • Toplam karakter: <b>{total_chars}</b></span></div>',
    unsafe_allow_html=True
)

preview_text = "\n".join(pages).strip()
if not preview_text:
    st.error("PDF içinden metin çıkarılamadı. (Metin katmanı yok / özel encoding). OCR gerekebilir.")
    st.stop()

with st.expander("Çıkan metin önizleme (ilk 300 karakter)"):
    st.code(preview_text[:300])

chunks = make_chunks(pages)
if not chunks:
    st.error("Metin çıkarıldı ama parçalama sonrası kullanılabilir içerik kalmadı. (Çok kısa veya dağınık içerik olabilir.)")
    st.stop()

with st.spinner("İndeks hazırlanıyor..."):
    index = build_index(chunks)

if index is None:
    st.error("İndeks oluşturulamadı (chunk sayısı 0).")
    st.stop()

st.success("PDF hazır. Sorunuzu yazabilirsiniz.")

# =========================
# CHAT
# =========================
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


