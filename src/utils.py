import re
import pandas as pd

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(inc|incorporated|corp|corporation|ltd|limited|pvt|private|llc|sarl|sa|sas|gmbh|eurl|co|company)\b',
    flags=re.IGNORECASE
)

def normalize_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = LEGAL_SUFFIXES_REGEX.sub("", text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

def extract_digits(text: str) -> set:
    if not isinstance(text, str) or pd.isna(text):
        return set()
    return set(re.findall(r'\b\d+\b', text))

def load_source_df(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", dtype=str).fillna("")
    df["business_name_norm"] = df["business_name"].apply(normalize_text)
    df["business_address_norm"] = df["business_address"].apply(normalize_text)
    df["digits_name"] = df["business_name"].apply(extract_digits)
    df["digits_address"] = df["business_address"].apply(extract_digits)
    df["country_norm"] = df["country"].str.strip().str.upper()
    return df
