import os
import argparse
import pandas as pd
import numpy as np

from utils import load_source_df
from blocking import generate_candidate_pairs
from features import extract_pairwise_features
from train_infer import train_lightgbm_model, format_predictions_tsv

def run_pipeline(train_dir: str, test_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/4] Loading Test Datasets...")
    s1_test = load_source_df(os.path.join(test_dir, "test_source1.tsv"))
    s2_test = load_source_df(os.path.join(test_dir, "test_source2.tsv"))
    s3_test = load_source_df(os.path.join(test_dir, "test_source3.tsv"))
    s23_test = pd.concat([s2_test, s3_test], ignore_index=True)

    print("[2/4] Running Candidate Blocking...")
    test_cands = generate_candidate_pairs(s1_test, s23_test, top_k=25, min_sim=0.12)
    
    cand_records = [{"source1_entity_id": k, "candidate_entity_ids": ",".join(v)} for k, v in test_cands.items()]
    pd.DataFrame(cand_records).to_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False)

    print("[3/4] Loading Train Data and Training LightGBM...")
    s1_train = load_source_df(os.path.join(train_dir, "train_source1.tsv"))
    s2_train = load_source_df(os.path.join(train_dir, "train_source2.tsv"))
    s3_train = load_source_df(os.path.join(train_dir, "train_source3.tsv"))
    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t").fillna("")

    gt_map = set()
    for _, row in gt_df.iterrows():
        for m_id in str(row["matched_entity_ids"]).split(","):
            if m_id.strip():
                gt_map.add((row["source1_entity_id"], m_id.strip()))

    train_cands = generate_candidate_pairs(s1_train, s23_train, top_k=25, min_sim=0.12)
    s1_tr_map = s1_train.set_index("entity_id")
    s23_tr_map = s23_train.set_index("entity_id")

    X_train, y_train = [], []
    for s1_id, c_ids in train_cands.items():
        if s1_id not in s1_tr_map.index:
            continue
        s1_row = s1_tr_map.loc[s1_id]
        for c_id in c_ids:
            if c_id not in s23_tr_map.index:
                continue
            X_train.append(extract_pairwise_features(s1_row, s23_tr_map.loc[c_id]))
            y_train.append(1 if (s1_id, c_id) in gt_map else 0)

    model = train_lightgbm_model(np.array(X_train), np.array(y_train))

    print("[4/4] Predicting Test Matches and Formatting Deliverables...")
    s1_ts_map = s1_test.set_index("entity_id")
    s23_ts_map = s23_test.set_index("entity_id")
    X_test, pair_records = [], []

    for s1_id, c_ids in test_cands.items():
        if s1_id not in s1_ts_map.index:
            continue
        s1_row = s1_ts_map.loc[s1_id]
        for c_id in c_ids:
            if c_id not in s23_ts_map.index:
                continue
            X_test.append(extract_pairwise_features(s1_row, s23_ts_map.loc[c_id]))
            pair_records.append({"s1_id": s1_id, "cand_id": c_id})

    probs = model.predict_proba(np.array(X_test))[:, 1] if X_test else np.array([])
    results_df = format_predictions_tsv(s1_test["entity_id"].tolist(), pd.DataFrame(pair_records), probs, threshold=0.78)
    results_df.to_csv(os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False)
    print("✅ SUCCESS! Deliverables saved to /content/output/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    run_pipeline(args.train_dir, args.test_dir, args.output_dir)
