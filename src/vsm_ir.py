import math
from collections import Counter

# --- Pre-computation ---

def calculate_tf(docs):
    """Menghitung Term Frequency (TF)."""
    tf = {}
    for doc_id, tokens in docs.items():
        tf[doc_id] = Counter(tokens)
    return tf

def calculate_df(docs):
    """Menghitung Document Frequency (DF)."""
    df = Counter()
    for tokens in docs.values():
        df.update(set(tokens))
    return df

def calculate_idf(df, N_docs):
    """Menghitung Inverse Document Frequency (IDF)."""
    idf = {}
    for term, doc_count in df.items():
        # Tambahkan 1 ke denominator untuk smoothing
        idf[term] = math.log10(N_docs / (doc_count + 1))
    return idf

# --- Vector & Similarity (DIMODIFIKASI) ---

def build_tfidf_matrix(tf, idf, scheme='sublinear_tf'):
    """
    Membangun TF-IDF Matriks (Sparse Representation).
    MODIFIKASI: Menambahkan 'scheme' untuk perbandingan (Soal 5.1).
    Skema: 'sublinear_tf' (1 + log(tf)) atau 'raw_tf' (tf).
    """
    tfidf_matrix = {}
    for doc_id, term_counts in tf.items():
        doc_vector = {}
        for term, count in term_counts.items():
            
            tf_weight = 0.0
            if scheme == 'sublinear_tf':
                # Skema 1: Sublinear TF (1 + log(tf))
                tf_weight = (1 + math.log10(count)) if count > 0 else 0
            elif scheme == 'raw_tf':
                # Skema 2: Raw TF (tf)
                tf_weight = count
            
            tfidf_score = tf_weight * idf.get(term, 0)
            if tfidf_score > 0:
                doc_vector[term] = tfidf_score
        tfidf_matrix[doc_id] = doc_vector
    return tfidf_matrix

def vectorize_query(query_tokens, idf, scheme='sublinear_tf'):
    """
    Representasi query sebagai vektor TF-IDF.
    MODIFIKASI: Menambahkan 'scheme' agar konsisten.
    """
    query_tf = Counter(query_tokens)
    query_vector = {}
    for term, count in query_tf.items():
        
        tf_weight = 0.0
        if scheme == 'sublinear_tf':
            tf_weight = (1 + math.log10(count)) if count > 0 else 0
        elif scheme == 'raw_tf':
            tf_weight = count
            
        tfidf_score = tf_weight * idf.get(term, 0)
        if tfidf_score > 0:
            query_vector[term] = tfidf_score
            
    return query_vector

def cosine_similarity(doc_vector, query_vector):
    """Menghitung Cosine Similarity."""
    
    dot_product = 0
    for term, q_weight in query_vector.items():
        d_weight = doc_vector.get(term, 0)
        dot_product += q_weight * d_weight
        
    if dot_product == 0:
        return 0.0

    doc_magnitude = math.sqrt(sum(w**2 for w in doc_vector.values()))
    query_magnitude = math.sqrt(sum(w**2 for w in query_vector.values()))
    
    if doc_magnitude == 0 or query_magnitude == 0:
        return 0.0
        
    return dot_product / (doc_magnitude * query_magnitude)

def rank_documents(tfidf_matrix, query_vector, k):
    """Menghitung similarity dan meranking dokumen."""
    rankings = []
    for doc_id, doc_vector in tfidf_matrix.items():
        score = cosine_similarity(doc_vector, query_vector)
        if score > 0:
            rankings.append((doc_id, score))
            
    rankings.sort(key=lambda item: item[1], reverse=True)
    return rankings[:k]