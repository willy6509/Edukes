import math
import sys
import os

# Menambahkan path src agar dapat mengimpor modul search
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Impor fungsi pencarian aktual dari modul Anda
try:
    from src.search import search_vsm, search_boolean
except ImportError:
    print("Error: Gagal mengimpor modul 'src.search'. Pastikan file ada dan benar.")
    sys.exit(1)

# --- 1. Definisi Gold Set (Qrels) ---
# (Soal 3 & 4 Uji Wajib)
# ANDA HARUS MENYESUAIKAN INI DENGAN 10 KORPUS ANDA
GOLD_SET = {
    "Q1": {
        "query": "cuci tangan sabun kuman",
        "relevant_docs_graded": {
            "doc01.txt": 1  # Sangat relevan
        }
    },
    "Q2": {
        "query": "kesehatan jantung dan gula",
        "relevant_docs_graded": {
            "doc04.txt": 2, # Bahas gula & jantung (Sangat Relevan)
            "doc03.txt": 1, # Pola hidup sehat (Relevan)
            "doc09.txt": 1  # Olahraga (Relevan)
        }
    },
    "Q3": {
        "query": "olahraga dan makanan sehat",
        "relevant_docs_graded": {
            "doc03.txt": 2, # Pilar hidup sehat (Sangat Relevan)
            "doc07.txt": 2, # Sarapan sehat (Sangat Relevan)
            "doc09.txt": 1  # Olahraga di rumah (Relevan)
        }
    }
}


# --- 2. Implementasi Metrik Matematika (Tidak Berubah) ---

def precision_recall_f1(retrieved_docs, relevant_docs):
    """Menghitung Precision, Recall, dan F1-Score (Soal 3 & 5.4a)."""
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = retrieved_set.intersection(relevant_set)
    
    precision = len(true_positives) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    recall = len(true_positives) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def average_precision_at_k(retrieved_docs, relevant_docs, k=10):
    """Menghitung Average Precision (AP@k) (Soal 4 & 5.4b)."""
    if not relevant_docs:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    
    sum_precisions = 0.0
    relevant_hits = 0
    
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_set:
            relevant_hits += 1
            precision_at_i = relevant_hits / (i + 1)
            sum_precisions += precision_at_i
            
    return sum_precisions / len(relevant_set)

def ndcg_at_k(retrieved_docs, relevant_docs_graded, k=10):
    """Menghitung nDCG@k (Soal 4 & 5.4b)."""
    retrieved_at_k = retrieved_docs[:k]
    
    # 1. Hitung DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        relevance = relevant_docs_graded.get(doc_id, 0)
        dcg += relevance / math.log2(i + 2) # i+2 karena i mulai dari 0
        
    # 2. Hitung IDCG@k
    ideal_relevance_scores = sorted(relevant_docs_graded.values(), reverse=True)
    
    idcg = 0.0
    for i in range(min(k, len(ideal_relevance_scores))):
        idcg += ideal_relevance_scores[i] / math.log2(i + 2)
        
    # 3. Hitung nDCG
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg


# --- 3. Orkestrasi Evaluasi (DIMODIFIKASI) ---

def run_evaluation():
    """Fungsi utama untuk menjalankan evaluasi berdasarkan GOLD_SET."""
    
    print("==============================================")
    print("🏁 MEMULAI EVALUASI SISTEM TEMU KEMBALI 🏁")
    print("==============================================")
    print(f"Menggunakan {len(GOLD_SET)} query dari GOLD_SET...\n")
    
    k = 10  # Set K global
    
    # --- A. Evaluasi Boolean Model (Soal 3) ---
    print("\n--- 1. Evaluasi Boolean Retrieval (Soal 3) ---")
    total_precision, total_recall, total_f1 = 0, 0, 0
    
    for q_id, data in GOLD_SET.items():
        query = data["query"]
        relevant_docs_set = set(data["relevant_docs_graded"].keys())
        
        retrieved_results = search_boolean(query)
        retrieved_doc_ids = [doc_id for doc_id, score, _ in retrieved_results]
        
        p, r, f1 = precision_recall_f1(retrieved_doc_ids, relevant_docs_set)
        
        print(f"  [Query: {query.ljust(25)}] -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        total_precision += p; total_recall += r; total_f1 += f1
        
    print(f"  [Rata-rata Boolean]           -> Avg P: {(total_precision/len(GOLD_SET)):.4f}, Avg R: {(total_recall/len(GOLD_SET)):.4f}, Avg F1: {(total_f1/len(GOLD_SET)):.4f}")


    # --- B. Evaluasi VSM (MODIFIKASI - Soal 4 & 5.4) ---
    print(f"\n--- 2. Evaluasi Vector Space Model (MAP@{k} & nDCG@{k}) ---")
    
    # Tentukan skema yang akan diuji (Soal 5.1)
    schemes_to_test = ['sublinear_tf', 'raw_tf']
    results_by_scheme = {}

    for scheme in schemes_to_test:
        print(f"\n  Menguji Skema: '{scheme}' ...")
        
        list_of_ap = []
        list_of_ndcg = []
        
        for q_id, data in GOLD_SET.items():
            query = data["query"]
            relevant_docs_graded = data["relevant_docs_graded"]
            relevant_docs_binary = {doc_id for doc_id, score in relevant_docs_graded.items() if score > 0}

            # Panggil search_vsm dengan skema yang benar
            retrieved_results = search_vsm(query, k=k, scheme=scheme)
            retrieved_doc_ids = [doc_id for doc_id, score, _ in retrieved_results]
            
            ap_score = average_precision_at_k(retrieved_doc_ids, relevant_docs_binary, k)
            list_of_ap.append(ap_score)
            
            ndcg_score = ndcg_at_k(retrieved_doc_ids, relevant_docs_graded, k)
            list_of_ndcg.append(ndcg_score)

        map_at_k = sum(list_of_ap) / len(list_of_ap) if list_of_ap else 0.0
        avg_ndcg_at_k = sum(list_of_ndcg) / len(list_of_ndcg) if list_of_ndcg else 0.0
        
        results_by_scheme[scheme] = {"MAP": map_at_k, "nDCG": avg_ndcg_at_k}
        print(f"  [Rata-rata {scheme}]           -> MAP@{k}: {map_at_k:.4f}, Avg nDCG@{k}: {avg_ndcg_at_k:.4f}")

    # --- C. Laporan Perbandingan (Soal 5.4) ---
    print("\n--- 3. Perbandingan Skema Bobot (Soal 5.4) ---")
    print("Tabel ini untuk Laporan.pdf")
    print(f"| {'Skema'.ljust(15)} | {'MAP@' + str(k):<10} | {'Avg nDCG@' + str(k):<10} |")
    print("|-" + "-"*15 + "-|-" + "-"*10 + "-|-" + "-"*10 + "-|")
    
    for scheme, metrics in results_by_scheme.items():
        print(f"| {scheme.ljust(15)} | {metrics['MAP']:<10.4f} | {metrics['nDCG']:<10.4f} |")

    print("\n==============================================")
    print("🏁 EVALUASI SELESAI 🏁")
    print("==============================================")


if __name__ == "__main__":
    print("Menjalankan modul evaluasi sebagai script utama...")
    print("Pastikan semua model telah dimuat oleh search.py...")
    run_evaluation()