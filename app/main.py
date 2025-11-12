import streamlit as st
import sys
import os
import time
import re  # Impor Regex untuk memecah kalimat

# Ambil path ke direktori 'app/'
current_dir = os.path.dirname(__file__)
# Ambil path ke direktori root proyek (satu level di atas 'app/')
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Tambahkan root proyek ke sys.path
sys.path.append(project_root)
# -------------------------------------

# Import search logic dari modul yang sudah dibuat
from src import search
from src import preprocess
from src import vsm_ir
from src import boolean_ir

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
    tf = vsm_ir.calculate_tf(docs_tokens)
    df = vsm_ir.calculate_df(docs_tokens)
    idf = vsm_ir.calculate_idf(df, N)
    
    # UI ini hanya akan menggunakan satu skema, misal 'sublinear_tf'
    tfidf_matrix = vsm_ir.build_tfidf_matrix(tf, idf, scheme='sublinear_tf')
    print("Data UI Streamlit (sublinear_tf) siap.")
    
    return docs_tokens, idf, tfidf_matrix

UI_DOCS_TOKENS, UI_IDF, UI_TFIDF_MATRIX = load_data()

# --- 3. Fungsi Utility (Termasuk Rangkuman Baru) ---

@st.cache_data # Cache hasil pembacaan file mentah
def get_raw_text(doc_id):
    """Membaca teks mentah dari file berdasarkan doc_id."""
    try:
        # Asumsi 'data' adalah folder di level yang sama dengan 'app'
        file_path = os.path.join(project_root, 'data/raw', doc_id)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Gagal membaca file {doc_id}: {e}")
        return ""

def split_into_sentences(text):
    """Memecah teks menjadi kalimat menggunakan regex sederhana."""
    # Memecah berdasarkan titik atau tanda tanya, diikuti spasi
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # Filter kalimat pendek/kosong
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def generate_extractive_summary(rankings, query_str, max_sentences=2):
    """
    Rangkuman Ekstraktif (Soal 5.3.b - Versi Baru).
    Mencari kalimat terbaik dari top-k dokumen berdasarkan query.
    """
    try:
        # 1. Dapatkan token query yang sudah diproses
        processed_query_tokens = set(preprocess.preprocess_document(query_str))
        if not processed_query_tokens:
            return "Query tidak valid untuk rangkuman."

        # 2. Kumpulkan kalimat dari 3 dokumen teratas
        all_sentences = []
        # Ambil top 3 doc_id dari hasil ranking
        doc_ids_to_check = [doc_id for doc_id, score, explain in rankings[:3]]
        
        for doc_id in doc_ids_to_check:
            raw_text = get_raw_text(doc_id)
            all_sentences.extend(split_into_sentences(raw_text))

        if not all_sentences:
            return "Tidak ada konten yang dapat dirangkum dari hasil teratas."

        # 3. Beri skor setiap kalimat berdasarkan overlap query
        scored_sentences = []
        for sentence in all_sentences:
            # Preprocess kalimat untuk perbandingan yang adil
            sentence_tokens = set(preprocess.preprocess_document(sentence))
            # Skor = jumlah token query yang ada di kalimat
            score = len(processed_query_tokens.intersection(sentence_tokens))
            
            if score > 0:
                scored_sentences.append((score, sentence))

        # 4. Urutkan berdasarkan skor dan ambil N kalimat teratas
        if not scored_sentences:
            # Jika tidak ada overlap, ambil cuplikan dari doc #1 (fallback)
            return get_snippet(rankings[0][0], max_char=200)

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        top_sentences = [s_text for s_score, s_text in scored_sentences[:max_sentences]]
        
        return " ".join(top_sentences)
    
    except Exception as e:
        print(f"Error di generate_extractive_summary: {e}")
        # Fallback jika terjadi error
        return get_snippet(rankings[0][0], max_char=200)


def get_snippet(doc_id, max_char=180):
    """Mendapatkan snippet dari raw document (sekarang sebagai fallback)."""
    raw_text = get_raw_text(doc_id)
    return raw_text.strip()[:max_char] + '...' if len(raw_text.strip()) > max_char else raw_text.strip()


def ui_search_vsm(query_str, k):
    """Fungsi VSM khusus untuk UI (memisahkan dari search.py)."""
    query_processed_tokens = preprocess.preprocess_document(query_str)
    query_vector = vsm_ir.vectorize_query(query_processed_tokens, UI_IDF, scheme='sublinear_tf')
    rankings = vsm_ir.rank_documents(UI_TFIDF_MATRIX, query_vector, k)
    
    # Tambahkan explainability
    explained_rankings = []
    query_terms_set = set(query_processed_tokens)
    for doc_id, score in rankings:
        doc_tokens_set = set(UI_DOCS_TOKENS.get(doc_id, [])) # Gunakan .get() agar aman
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
        # --- Rangkuman Cepat (LOGIKA BARU) ---
        with st.spinner("Membuat rangkuman cepat..."):
            # Panggil fungsi rangkuman ekstraktif yang baru
            summary_text = generate_extractive_summary(rankings, st.session_state.current_query, max_sentences=2)
        st.info(f"**Rangkuman Cepat:** {summary_text}")
        
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