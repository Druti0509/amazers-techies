import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_candidate_pairs(s1_df: pd.DataFrame, s23_df: pd.DataFrame, top_k: int = 25, min_sim: float = 0.12, batch_size: int = 2000):
    s1_texts = (s1_df["business_name_norm"] + " " + s1_df["business_address_norm"]).tolist()
    s23_texts = (s23_df["business_name_norm"] + " " + s23_df["business_address_norm"]).tolist()
    s1_ids = s1_df["entity_id"].tolist()
    s23_ids = s23_df["entity_id"].values

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=2)
    vectorizer.fit(s23_texts)
    X_s23 = vectorizer.transform(s23_texts)
    candidates_dict = {}

    for i in range(0, len(s1_texts), batch_size):
        batch_texts = s1_texts[i:i + batch_size]
        batch_ids = s1_ids[i:i + batch_size]
        X_batch = vectorizer.transform(batch_texts)
        sim_matrix = X_batch.dot(X_s23.T)

        for row_idx, s1_id in enumerate(batch_ids):
            row = sim_matrix[row_idx]
            if row.nnz == 0:
                candidates_dict[s1_id] = []
                continue
            indices, data = row.indices, row.data
            valid_mask = data >= min_sim
            if not np.any(valid_mask):
                candidates_dict[s1_id] = []
                continue
            valid_indices, valid_scores = indices[valid_mask], data[valid_mask]
            top_order = np.argsort(valid_scores)[::-1][:top_k]
            candidates_dict[s1_id] = list(dict.fromkeys(s23_ids[valid_indices[top_order]].tolist()))

    return candidates_dict
