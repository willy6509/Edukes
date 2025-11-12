import streamlit as st
import sys
import os
import time

# Tambahkan src ke path agar modul bisa diimpor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import search logic dari modul yang sudah dibuat
from src import search
from src import preprocess
from src import vsm_ir  # <-- TAMBAHKAN BARIS INI
from src import boolean_ir # <-- Tambahkan juga ini untuk jaga-jaga

# --- 1. Konfigurasi Halaman & Styling Kustom ---

st.set_page_config(
    page_title="EduKesehatan Search Engine",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Styling CSS Kustom (Sadar-Tema & Responsif)
custom_css = """
<style>
/* ... (Seluruh CSS dari respons terakhir Anda tetap sama) ... */
body, .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.header {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
    padding: 10px 0;
    margin-bottom: 2rem;
    text-align: center;
}
.stTextInput > div > div > input {
    border-radius: 50px;
    padding: 14px 22px;
    font-size: 1.1rem;
    background-color: var(--secondary-background-color);
    border: 1px solid var(--border-color, #ddd);
    color: var(--text-color);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px var(--primary-color);
    opacity: 0.7;
}
.stButton>button {
    border-radius: 50px;
    font-weight: 600;
    padding: 14px 25px;
    width: 100%;
    background-color: var(--primary-color);
    color: white;
    border: none;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton>button:hover {
    opacity: 0.8;
    transform: translateY(-2px);
}
@media (max-width: 768px) {
    .header { font-size: 1.25rem; margin-bottom: 1rem; }
    .stTextInput > div > div > input { padding: 10px 18px; font-size: 1rem; }
    .stButton>button { padding: 12px 20px; }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. Load Data (Cache) ---

@st.cache_resource
def load_data():
    """Load semua data/index saat aplikasi dimulai dan cache."""
    print("Memuat data untuk UI Streamlit...")
    processed_docs = preprocess.load_documents('data/processed')
    docs_tokens = {doc_id: preprocess.tokenize(text) for doc_id, text in processed_docs.items()}
    
    N = len(docs_tokens)
    # vsm_ir sekarang sudah terdefinisi berkat impor di atas
    tf = vsm_ir.calculate_tf(docs_tokens)
    df = vsm_ir.calculate_df(docs_tokens)
    idf = vsm_ir.calculate_idf(df, N)
    
    # UI ini hanya akan menggunakan satu skema, misal 'sublinear_tf'
    tfidf_matrix = vsm_ir.build_tfidf_matrix(tf, idf, scheme='sublinear_tf')
    print("Data UI Streamlit (sublinear_tf) siap.")
    
    return docs_tokens, idf, tfidf_matrix

# Kita memuat data VSM secara mandiri untuk UI
UI_DOCS_TOKENS, UI_IDF, UI_TFIDF_MATRIX = load_data()

# --- 3. Fungsi Utility ---

def get_snippet(doc_id, max_char=180):
    """Mendapatkan snippet dari raw document."""
    try:
        raw_docs = preprocess.load_documents('data/raw')
        text = raw_docs.get(doc_id, "Dokumen tidak ditemukan.")
        return text.strip()[:max_char] + '...' if len(text.strip()) > max_char else text.strip()
    except Exception:
        return "Gagal memuat snippet."

def ui_search_vsm(query_str, k):
    """Fungsi VSM khusus untuk UI (memisahkan dari search.py)."""
    
    query_processed_tokens = preprocess.preprocess_document(query_str)
    # vsm_ir sekarang sudah terdefinisi
    query_vector = vsm_ir.vectorize_query(query_processed_tokens, UI_IDF, scheme='sublinear_tf')
    rankings = vsm_ir.rank_documents(UI_TFIDF_MATRIX, query_vector, k)
    
    # Tambahkan explainability
    explained_rankings = []
    query_terms_set = set(query_processed_tokens)
    for doc_id, score in rankings:
        doc_tokens_set = set(UI_DOCS_TOKENS[doc_id])
        matching_terms = list(query_terms_set.intersection(doc_tokens_set))
        explained_rankings.append((doc_id, score, matching_terms[:5]))
        
    return explained_rankings

# --- 4. Streamlit UI Layout (DIMODIFIKASI) ---

if 'rankings' not in st.session_state:
    st.session_state.rankings = None

st.markdown("<div class='header'>🧠 EduKesehatan Search Engine</div>", unsafe_allow_html=True)

with st.container():
    query = st.text_input(
        "Masukkan Query Pencarian",
        placeholder="Contoh: Manfaat air putih dan olahraga...",
        key="current_query",
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        k_val = st.slider("Jumlah Hasil (K)", min_value=1, max_value=10, value=5)
    with col2:
        search_pressed = st.button("🔎 Lakukan Pencarian")

    if search_pressed:
        if st.session_state.current_query:
            with st.spinner('Menganalisis dan meranking dokumen...'):
                time.sleep(1) # Simulasi jeda
                st.session_state.rankings = ui_search_vsm(st.session_state.current_query, k_val)
        else:
            st.error("Mohon masukkan query pencarian terlebih dahulu.")
            st.session_state.rankings = None

if st.session_state.rankings is not None:
    rankings = st.session_state.rankings
    st.markdown("---")

    if rankings:
        # Rangkuman Cepat (Soal 5.3)
        top_doc_id, top_score, top_explain = rankings[0]
        top_snippet = get_snippet(top_doc_id, max_char=200)
        st.info(f"**Rangkuman Cepat:** Hasil teratas ({top_doc_id}) menyarankan: *\"{top_snippet}\"*")
        
        st.subheader(f"📚 Hasil Pencarian Detil (Top {len(rankings)})")
        
        for rank, (doc_id, score, explain_terms) in enumerate(rankings):
            snippet = get_snippet(doc_id)
            
            with st.container(border=True):
                st.caption(f"Sumber Dokumen: {doc_id}")
                st.markdown(f"**#{rank + 1}** — Relevansi (Skor): **{score:.4f}**")
                st.write(snippet)
                if explain_terms:
                    explain_str = ", ".join(explain_terms)
                    st.caption(f"Istilah Cocok: {explain_str}")
    else:
        st.warning("Tidak ditemukan dokumen yang relevan. Coba ganti kata kunci Anda.")