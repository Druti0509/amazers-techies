import lightgbm as lgb
import numpy as np
import pandas as pd

def train_lightgbm_model(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    return model

def format_predictions_tsv(s1_ids: list, candidate_pairs_df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.78) -> pd.DataFrame:
    candidate_pairs_df["prob"] = probs
    matched_results = []
    grouped = candidate_pairs_df.groupby("s1_id")

    for s1_id in s1_ids:
        if s1_id not in grouped.groups:
            matched_results.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
            continue
        sub_df = grouped.get_group(s1_id)
        valid_matches = sub_df[sub_df["prob"] >= threshold].sort_values(by="prob", ascending=False)
        matched_ids_str = ",".join(list(dict.fromkeys(valid_matches["cand_id"].tolist()))) if not valid_matches.empty else ""
        matched_results.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_ids_str})

    return pd.DataFrame(matched_results)
