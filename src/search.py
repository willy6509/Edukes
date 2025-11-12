import sys
import argparse
import os

# Menambahkan path agar bisa impor modul dari root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import preprocess, boolean_ir, vsm_ir

# --- Setup Global (MODIFIKASI) ---
def load_all_data(doc_dir='data/processed'):
    """Memuat semua indeks dan model yang diperlukan saat startup."""
    print("Memuat dokumen terproses...")
    processed_docs = preprocess.load_documents(doc_dir)
    docs_tokens = {doc_id: preprocess.tokenize(text) for doc_id, text in processed_docs.items()}
    
    print("Membangun Indeks Boolean...")
    inverted_index = boolean_ir.build_inverted_index(docs_tokens)
    
    print("Membangun komponen VSM (TF, DF, IDF)...")
    N = len(docs_tokens)
    tf = vsm_ir.calculate_tf(docs_tokens)
    df = vsm_ir.calculate_df(docs_tokens)
    idf = vsm_ir.calculate_idf(df, N)
    
    # MODIFIKASI: Buat 2 Matriks TF-IDF (Soal 5.1)
    print("Pre-computing TF-IDF Matrix (Scheme: sublinear_tf)...")
    tfidf_matrix_sublinear = vsm_ir.build_tfidf_matrix(tf, idf, scheme='sublinear_tf')
    
    print("Pre-computing TF-IDF Matrix (Scheme: raw_tf)...")
    tfidf_matrix_raw = vsm_ir.build_tfidf_matrix(tf, idf, scheme='raw_tf')
    
    print("Semua model siap.")
    return docs_tokens, inverted_index, idf, tfidf_matrix_sublinear, tfidf_matrix_raw

# Muat semua model saat startup
DOCS_TOKENS, INVERTED_INDEX, IDF, TFIDF_MATRIX_SUBLINEAR, TFIDF_MATRIX_RAW = load_all_data()
ALL_DOC_IDS = set(DOCS_TOKENS.keys())

# --- Core Search Logic (MODIFIKASI) ---

def search_boolean(query_str):
    """Search menggunakan Boolean Model."""
    results = boolean_ir.parse_and_execute_boolean_query(query_str, INVERTED_INDEX, ALL_DOC_IDS)
    # (Explainability Boolean bisa ditambahkan di sini jika perlu)
    return [(doc_id, 1.0, []) for doc_id in results] # Tambah list kosong untuk konsistensi

def search_vsm(query_str, k, scheme='sublinear_tf'):
    """Search menggunakan VSM (MODIFIKASI: memilih skema dan menambah explain)."""
    
    # Pilih matriks yang sesuai
    tfidf_matrix = TFIDF_MATRIX_SUBLINEAR if scheme == 'sublinear_tf' else TFIDF_MATRIX_RAW

    query_processed_tokens = preprocess.preprocess_document(query_str)
    query_vector = vsm_ir.vectorize_query(query_processed_tokens, IDF, scheme=scheme)
    rankings = vsm_ir.rank_documents(tfidf_matrix, query_vector, k)
    
    # MODIFIKASI: Tambahkan data 'explain' (Soal 3 & 5.2)
    explained_rankings = []
    query_terms_set = set(query_processed_tokens)
    for doc_id, score in rankings:
        doc_tokens_set = set(DOCS_TOKENS[doc_id])
        # Cari irisan antara token query dan token dokumen
        matching_terms = list(query_terms_set.intersection(doc_tokens_set))
        explained_rankings.append((doc_id, score, matching_terms[:5])) # Ambil 5 top term
        
    return explained_rankings

# --- CLI Interface (MODIFIKASI) ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mini Search Engine EduKesehatan CLI. (Soal 5.2)")
    parser.add_argument('--model', choices=['boolean', 'vsm'], required=True, help="Model pencarian: boolean atau vsm.")
    parser.add_argument('--scheme', choices=['sublinear_tf', 'raw_tf'], default='sublinear_tf', help="Skema TF-IDF untuk VSM (Soal 5.1).")
    parser.add_argument('--k', type=int, default=5, help="Jumlah top dokumen untuk VSM.")
    parser.add_argument('--query', required=True, help="Query pencarian (gunakan tanda kutip).")
    
    args = parser.parse_args()
    
    results = []
    if args.model == 'boolean':
        print(f"\n--- Hasil Boolean Retrieval ---")
        results = search_boolean(args.query)
            
    elif args.model == 'vsm':
        print(f"\n--- Hasil VSM Retrieval (Top-{args.k}, Scheme: {args.scheme}) ---")
        results = search_vsm(args.query, args.k, args.scheme)
    
    # Cetak hasil
    if results:
        for doc_id, score, explain_terms in results:
            explain_str = ""
            if explain_terms:
                explain_str = f"| Explain (Istilah Cocok): {', '.join(explain_terms)}" # (Soal 3 & 5.2)
            
            print(f"-> {doc_id.ljust(15)} | Skor: {score:<8.4f} {explain_str}")
    else:
        print("Tidak ada dokumen yang relevan.")