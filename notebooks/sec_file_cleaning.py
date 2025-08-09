# Parsing and extracting SEC file data

import os
import re
import glob
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import textstat

nltk.download('punkt_tab')
nltk.download('stopwords')

# --- Load Loughran-McDonald financial sentiment dictionary ---
def load_lm_dicts(lm_dict_path):
    lm_df = pd.read_csv(lm_dict_path)
    lm_negative = set(lm_df[lm_df['Negative'] > 0]['Word'].str.lower())
    lm_positive = set(lm_df[lm_df['Positive'] > 0]['Word'].str.lower())
    return lm_negative, lm_positive

def extract_filing_date_from_txt(raw_txt):
    # Look for the "FILED AS OF DATE:" line and extract date
    match = re.search(r'FILED AS OF DATE:\s+(\d{8})', raw_txt)
    if match:
        # Format as YYYY-MM-DD
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return ""  # fallback if not found

def clean_text(text):
    # Remove HTML tags, non-letters, numbers, extra whitespace
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.strip()

def compute_lm_sentiment(tokens, neg_words, pos_words):
    tokens = [t.lower() for t in tokens if len(t) > 1]
    n_pos = sum(1 for t in tokens if t in pos_words)
    n_neg = sum(1 for t in tokens if t in neg_words)
    if n_pos + n_neg == 0:
        return 0
    return (n_pos - n_neg) / (n_pos + n_neg)

def extract_date_and_type(filename):
    # Try to extract date (YYYY-MM-DD or YYYYMMDD) and form type from filename or path
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if not date_match:
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            # Convert YYYYMMDD to YYYY-MM-DD
            date_str = date_match.group(1)
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            date = ""
    else:
        date = date_match.group(1)
    # Form type: look for 10-K, 10-Q, 8-K in path or filename
    type_match = re.search(r'(10[-\s]?K|10[-\s]?Q|8[-\s]?K)', filename, re.IGNORECASE)
    form_type = type_match.group(1).upper().replace(' ', '').replace('-', '') if type_match else ""
    return date, form_type

def process_filing(filepath, neg_words, pos_words):
    basename = os.path.basename(filepath)
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    filing_date = extract_filing_date_from_txt(raw)
    form_type = extract_date_and_type(filepath)[1]
    text = clean_text(raw)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.lower() not in stopwords.words('english')]
    try:
        fog = textstat.gunning_fog(text)
    except Exception:
        fog = None
    sentiment = compute_lm_sentiment(tokens, neg_words, pos_words)
    return {
        'date': filing_date,
        'form_type': form_type,
        'filename': basename,
        'fog_index': fog,
        'sentiment': sentiment,
    }

def process_all_filings(folder, lm_dict_path, out_csv):
    neg_words, pos_words = load_lm_dicts(lm_dict_path)
    results = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.txt'):
                filepath = os.path.join(root, fname)
                summary = process_filing(filepath, neg_words, pos_words)
                results.append(summary)
    df = pd.DataFrame(results)
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved SEC filings features to {out_csv}")
    return df


if __name__ == "__main__":
    SEC_FOLDER = "../data/sec-edgar-filings/"
    LM_DICT_PATH = "../data/Loughran-McDonald_MasterDictionary_1993-2024.csv"
    OUT_CSV = "../data/sec_filings_features.csv"
    process_all_filings(SEC_FOLDER, LM_DICT_PATH, OUT_CSV)