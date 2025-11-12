import sys
import os

# Menambahkan path agar bisa impor modul dari root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import preprocess # Diperlukan untuk memuat dokumen

"""
Modul ini berisi implementasi untuk Soal 03: Boolean Retrieval Model.
Termasuk:
1. Pembangunan Vocabulary (implisit)
2. build_incidence_matrix (Soal 2a)
3. build_inverted_index (Soal 2b)
4. parse_and_execute_boolean_query (Soal 3)
"""

def build_vocabulary(docs_tokens):
    """
    Membangun vocabulary unik dari semua dokumen.
    (Langkah 1 Soal 03)
    
    :param docs_tokens: Dict[str, List[str]], misal {"doc01.txt": ["cuci", "tangan", ...]}
    :return: Set[str], vocabulary unik
    """
    vocabulary = set()
    for tokens in docs_tokens.values():
        vocabulary.update(tokens)
    return vocabulary

def build_incidence_matrix(docs_tokens):
    """
    Membangun Incidence Matrix (direpresentasikan sebagai sparse dict).
    (Langkah 2a Soal 03)
    
    :param docs_tokens: Dict[str, List[str]]
    :return: Tuple (matrix, term_map, doc_map)
             matrix: Dict[int, List[int]] -> {term_idx: [doc_idx1, doc_idx2]}
             term_map: Dict[str, int] -> {"cuci": 0, "tangan": 1, ...}
             doc_map: Dict[str, int] -> {"doc01.txt": 0, ...}
    """
    print("Membangun Incidence Matrix (Sparse)...")
    vocabulary = sorted(list(build_vocabulary(docs_tokens)))
    doc_ids = sorted(docs_tokens.keys())
    
    # Buat pemetaan dari string ke integer index
    term_map = {term: i for i, term in enumerate(vocabulary)}
    doc_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Matriks sparse: {term_index: [doc_index1, doc_index2, ...]}
    incidence_matrix_sparse = {i: [] for i in range(len(vocabulary))}
    
    for doc_id, tokens in docs_tokens.items():
        doc_idx = doc_map[doc_id]
        unique_tokens = set(tokens) # Matriks insiden hanya peduli 0 atau 1
        
        for token in unique_tokens:
            if token in term_map:
                term_idx = term_map[token]
                incidence_matrix_sparse[term_idx].append(doc_idx)
                
    return incidence_matrix_sparse, term_map, doc_map

def build_inverted_index(docs_tokens):
    """
    Membangun Inverted Index sederhana (dict(term -> postings)).
    (Langkah 2b Soal 03)
    
    :param docs_tokens: Dict[str, List[str]]
    :return: Dict[str, Set[str]] -> {"cuci": {"doc01.txt", "doc02.txt"}, ...}
    """
    print("Membangun Inverted Index...")
    inverted_index = {}
    for doc_id, tokens in docs_tokens.items():
        for token in set(tokens): # Gunakan set untuk token unik di dokumen
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)
            
    return inverted_index

def parse_and_execute_boolean_query(query_str, index, all_doc_ids):
    """
    Parser Query Boolean sederhana: mendukung AND, OR, NOT.
    (Langkah 3 Soal 03)
    
    Implementasi ini adalah parser linear sederhana dan tidak mendukung kurung.
    Asumsi: 'term1 OPERATOR term2 OPERATOR term3 ...'
    Contoh: "sehat AND olahraga NOT gula"
    
    :param query_str: String query, misal "cuci AND tangan OR sabun"
    :param index: Inverted Index (Dict[str, Set[str]])
    :param all_doc_ids: Set[str] dari semua ID dokumen (untuk operasi NOT)
    :return: List[str] dari doc_id yang cocok (diurutkan)
    """
    
    # Pre-process query: sama seperti dokumen
    query_tokens_raw = preprocess.preprocess_document(query_str)
    
    # Pisahkan operator dari term
    terms = []
    operators = []
    
    for token in query_tokens_raw:
        if token in ['and', 'or', 'not']:
            operators.append(token.upper())
        else:
            terms.append(token)
            
    if not terms:
        return []

    # Ambil postings list untuk term pertama
    current_result_set = index.get(terms[0], set()).copy()

    # Proses sisa query secara linear
    # i digunakan untuk mengakses 'terms' (mulai dari term kedua)
    # operator_idx digunakan untuk mengakses 'operators'
    
    operator_idx = 0
    i = 1
    
    while operator_idx < len(operators) and i < len(terms):
        op = operators[operator_idx]
        term = terms[i]
        
        term_postings = index.get(term, set()).copy()
        
        if op == 'AND':
            current_result_set.intersection_update(term_postings)
        
        elif op == 'OR':
            current_result_set.update(term_postings)
            
        elif op == 'NOT':
            # 'NOT term' berarti 'ambil semua DOKUMEN LAINNYA selain term'
            # Ini jarang digunakan sendirian, biasanya 'A AND NOT B'
            # Untuk parser sederhana ini, kita asumsikan 'CURRENT_SET NOT TERM'
            current_result_set.difference_update(term_postings)
        
        operator_idx += 1
        i += 1

    return sorted(list(current_result_set))


# --- Bagian Eksekusi (untuk pengujian mandiri) ---
if __name__ == "__main__":
    # Ini hanya akan berjalan jika Anda menjalankan: python src/boolean_ir.py
    print("Menjalankan tes mandiri untuk boolean_ir.py...")
    
    # 1. Muat dokumen (pastikan preprocess.py sudah dijalankan)
    try:
        processed_docs = preprocess.load_documents('data/processed')
        docs_tokens = {doc_id: preprocess.tokenize(text) for doc_id, text in processed_docs.items()}
        all_docs = set(docs_tokens.keys())
    except FileNotFoundError:
        print("\nERROR: Folder 'data/processed' tidak ditemukan.")
        print("Silakan jalankan 'python src/preprocess.py' terlebih dahulu.")
        sys.exit(1)

    # 2. Tes Pembangunan Indeks (Soal 2a & 2b)
    matrix, t_map, d_map = build_incidence_matrix(docs_tokens)
    index = build_inverted_index(docs_tokens)
    
    print(f"\nVocabulary Size: {len(index)}")
    print(f"Total Documents: {len(all_docs)}")
    print("Contoh Inverted Index ('tangan'):", index.get('tangan', 'Tidak ditemukan'))
    
    # 3. Tes Parser Query (Soal 3)
    q1 = "tangan dan sabun" # AND
    q2 = "jantung atau gula"  # OR
    q3 = "sehat dan olahraga bukan gula" # AND NOT
    
    print("\n--- Tes Query (Soal 3) ---")
    
    results1 = parse_and_execute_boolean_query(q1, index, all_docs)
    print(f"Hasil Query '{q1}': {results1}")
    
    results2 = parse_and_execute_boolean_query(q2, index, all_docs)
    print(f"Hasil Query '{q2}': {results2}")
    
    results3 = parse_and_execute_boolean_query(q3, index, all_docs)
    print(f"Hasil Query '{q3}': {results3}")