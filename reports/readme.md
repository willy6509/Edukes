# Proyek UTS STKI: Mini Search Engine EduKesehatan
**Nama:** (Nama Anda)
**NIM:** (NIM Anda)
**Kelas:** A11.4703

Proyek ini adalah implementasi *mini search engine* untuk artikel kesehatan dasar, dibangun untuk memenuhi Ujian Tengah Semester (UTS) Mata Kuliah Sistem Temu Kembali Informasi.

## 📁 Struktur Folder

```
stki-uts-<nim>-<nama>/
├── data/
│   ├── raw/
│   │   └── (10 .txt korpus)
│   └── processed/
│       └── (10 .txt korpus terproses)
├── src/
│   ├── preprocess.py      # (Soal 02) Modul preprocessing
│   ├── boolean_ir.py      # (Soal 03) Modul Boolean Retrieval
│   ├── vsm_ir.py          # (Soal 04) Modul Vector Space Model
│   ├── search.py          # (Soal 05) Orchestrator & CLI
│   └── eval.py            # (Soal 05) Skrip evaluasi (P/R/F1, MAP, nDCG)
├── app/
│   └── main.py            # (Soal 05) Antarmuka web Streamlit
├── notebooks/
│   └── UTS_STKI_<nim>.ipynb # (Soal 2,3,4,5) Analisis & Laporan Uji
├── reports/
│   ├── laporan.pdf        # (Wajib) Laporan analisis proyek
│   └── statistics.json    # (Otomatis) Output Uji Soal 2
├── readme.md              # (File ini)
└── requirements.txt       # Kebutuhan library Python
```

## 🚀 Cara Menjalankan Proyek

### A. Instalasi
Pastikan Anda memiliki Python 3.8+ dan `pip`.

1.  **Instal *Dependencies***:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Unduh *Resource* NLTK** (hanya sekali):
    Jalankan Python dan ketik:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

### B. Tahap 1: Preprocessing & Uji Soal 2
Jalankan *script* ini untuk membersihkan korpus di `data/raw/` dan menyimpan hasilnya di `data/processed/`. Perintah ini juga akan menjalankan **Uji Soal 2** (statistik token & panjang dokumen) dan menyimpan hasilnya di `reports/statistics.json`.

```bash
python src/preprocess.py
```

### C. Tahap 2: Menjalankan Antarmuka Web (Streamlit)
Ini adalah antarmuka utama proyek (Soal 5.3).

```bash
python -m streamlit run app/main.py
```
Buka browser Anda di `http://localhost:8501`.

### D. Tahap 3: Menjalankan Evaluasi Model (CLI)
*Script* ini akan menjalankan **Uji Wajib Soal 3** (P/R/F1 Boolean) dan **Uji Wajib Soal 4/5** (Perbandingan skema VSM) menggunakan `GOLD_SET`.

```bash
python src/eval.py
```

### E. Tahap 4: Melihat Analisis & Grafik (Notebook)
Untuk melihat dokumentasi proses, visualisasi, dan hasil Uji secara interaktif (Soal 2, 3, 4, 5).

```bash
# Pastikan Anda sudah menginstal jupyter
pip install jupyterlab

# Jalankan Jupyter
jupyter lab notebooks/UTS_STKI_<nim>.ipynb
```

## 🧐 Asumsi Implementasi
1.  **Preprocessing**: Menggunakan `NLTK` untuk *stopwords* dan `Sastrawi` untuk *stemming* Bahasa Indonesia.
2.  **Boolean Query**: Parser di `boolean_ir.py` hanya mendukung `AND`, `OR`, `NOT` tanpa tanda kurung `()`.
3.  **Perbandingan Skema**: Implementasi VSM mendukung 2 skema: `sublinear_tf` (default) dan `raw_tf` untuk perbandingan (Soal 5.1).
4.  **Gold Set**: *Truth set* untuk evaluasi didefinisikan secara manual di dalam `src/eval.py`.