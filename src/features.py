import pandas as pd
from rapidfuzz import fuzz

def compute_digit_jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))

def extract_pairwise_features(s1_row: pd.Series, cand_row: pd.Series) -> list:
    f1 = fuzz.ratio(s1_row["business_name_norm"], cand_row["business_name_norm"]) / 100.0
    f2 = fuzz.partial_ratio(s1_row["business_name_norm"], cand_row["business_name_norm"]) / 100.0
    f3 = fuzz.token_sort_ratio(s1_row["business_name_norm"], cand_row["business_name_norm"]) / 100.0
    f4 = fuzz.token_set_ratio(s1_row["business_name_norm"], cand_row["business_name_norm"]) / 100.0
    f5 = fuzz.ratio(s1_row["business_address_norm"], cand_row["business_address_norm"]) / 100.0
    f6 = fuzz.token_set_ratio(s1_row["business_address_norm"], cand_row["business_address_norm"]) / 100.0
    f7 = compute_digit_jaccard(
        s1_row["digits_name"].union(s1_row["digits_address"]),
        cand_row["digits_name"].union(cand_row["digits_address"])
    )
    f8 = 1.0 if (s1_row["country_norm"] != "" and s1_row["country_norm"] == cand_row["country_norm"]) else 0.0
    return [f1, f2, f3, f4, f5, f6, f7, f8]
