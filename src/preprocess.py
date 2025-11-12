import re
import os
import json
from collections import Counter
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Setup ---
STOPWORDS_ID = set(stopwords.words('indonesian'))
STEMMER = StemmerFactory().create_stemmer()

# --- Fungsi Inti (Tidak Berubah) ---

def load_documents(doc_dir='data/raw'):
    """Memuat semua dokumen dari folder."""
    docs = {}
    for filename in os.listdir(doc_dir):
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(doc_dir, filename), 'r', encoding='utf-8') as f:
                    docs[filename] = f.read()
            except Exception as e:
                print(f"Gagal memuat {filename}: {e}")
    return docs

def clean_text(text):
    """Case Folding dan Normalisasi Angka/Tanda Baca (ringkas)."""
    text = text.lower() # Case Folding
    text = re.sub(r'[^a-z\s]', ' ', text) # Hapus selain huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

def tokenize(text):
    """Tokenisasi sederhana."""
    return text.split()

def remove_stopwords(tokens):
    """Menghapus stop words."""
    return [token for token in tokens if token not in STOPWORDS_ID]

def stem(tokens):
    """Stemming menggunakan Sastrawi."""
    text = ' '.join(tokens)
    stemmed_text = STEMMER.stem(text)
    return stemmed_text.split()

def preprocess_document(text):
    """Fungsi orkestrasi preprocessing."""
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem(filtered_tokens)
    return stemmed_tokens

# --- Fungsi Baru (Untuk Uji Soal 2) ---

def get_doc_statistics(processed_docs_tokens):
    """
    Menghitung 10 token tersering dan panjang dokumen.
    (Memenuhi Uji Soal 2)
    """
    stats = {
        "doc_lengths": {},
        "top_10_tokens": {}
    }
    
    all_lengths = []
    
    for doc_id, tokens in processed_docs_tokens.items():
        # 1. Hitung panjang dokumen
        length = len(tokens)
        stats["doc_lengths"][doc_id] = length
        all_lengths.append(length)
        
        # 2. Hitung 10 token tersering
        token_counts = Counter(tokens)
        stats["top_10_tokens"][doc_id] = token_counts.most_common(10)
        
    # 3. Hitung distribusi panjang dokumen (data untuk grafik)
    if all_lengths:
        stats["distribution"] = {
            "mean": sum(all_lengths) / len(all_lengths),
            "min": min(all_lengths),
            "max": max(all_lengths),
        }
    
    return stats

# --- Bagian Eksekusi Utama (Diubah Total) ---

if __name__ == '__main__':
    # 1. Pastikan folder output ada
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # 2. Muat dan Proses Dokumen
    raw_docs = load_documents('data/raw')
    processed_docs_tokens = {} # Tampung token untuk statistik

    print("--- Memulai Preprocessing Dokumen ---")
    for doc_id, text in raw_docs.items():
        processed_tokens = preprocess_document(text)
        processed_docs_tokens[doc_id] = processed_tokens # Simpan token
        
        # Simpan hasil pemrosesan ke data/processed/
        processed_text = ' '.join(processed_tokens)
        output_filename = os.path.join('data/processed', doc_id)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(processed_text)
            
    print(f"--- Preprocessing Selesai ({len(raw_docs)} dokumen) ---")

    # 3. Jalankan Uji Soal 2
    print("\n--- Menjalankan Uji Soal 2 (Statistik Dokumen) ---")
    statistics = get_doc_statistics(processed_docs_tokens)
    
    # Simpan statistik ke file JSON untuk Laporan
    stats_path = 'reports/statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)
        
    print(f"Statistik (Top 10 Tokens & Panjang Dokumen) disimpan di {stats_path}")
    
    # Tampilkan sampel statistik di console
    print("\nContoh Top 10 Tokens (doc01.txt):")
    print(statistics.get("top_10_tokens", {}).get("doc01.txt", "doc01.txt tidak ditemukan"))
    
    print("\nDistribusi Panjang Dokumen (Data untuk Grafik):")
    print(statistics.get("distribution", "Data tidak tersedia"))