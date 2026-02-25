import os
import pandas as pd
import tldextract
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Config - paths to your Excel datasets (present in project root)
DATA_FILES = [
    "data_bal - 20000.xlsx",
    "data_imbal - 55000.xlsx",
]

# Labels: 0 = Legitimate, 1 = Phishing

SCAM_KEYWORDS = [
    "gcash", "paymaya", "bank", "send money", "transfer",
    "win", "winner", "congratulations", "prize", "reward", "free", "promo",
    "urgent", "verify", "login", "confirm",
    "investment", "double your money", "easy profit", "job", "work from home", "shop now", "limited offer", "act fast"
]


def load_datasets(files=DATA_FILES):
    dfs = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_excel(f)
            except Exception:
                df = pd.read_csv(f)
            dfs.append(df)
        else:
            print(f"Warning: dataset file not found: {f}")
    if not dfs:
        raise FileNotFoundError("No dataset files found. Put Excel/CSV files in project root.")
    data = pd.concat(dfs, ignore_index=True)
    # normalize common column names
    cols = {c: c.strip() for c in data.columns}
    data.rename(columns=cols, inplace=True)
    # Map common variants to expected names
    col_map = {}
    if 'urls' in [c.lower() for c in data.columns]:
        # find exact column name with case
        real = next(c for c in data.columns if c.lower()=='urls')
        col_map[real] = 'url'
    if 'labels' in [c.lower() for c in data.columns]:
        real = next(c for c in data.columns if c.lower()=='labels')
        col_map[real] = 'label'
    data = data.rename(columns=col_map)
    return data


def extract_features(url: str):
    url = str(url or "")
    parsed = urlparse(url)
    te = tldextract.extract(url)
    subdomain = te.subdomain or ""
    domain = te.domain or ""
    suffix = te.suffix or ""

    features = {
        "len_url": len(url),
        "count_digits": sum(c.isdigit() for c in url),
        "count_hyphen": url.count('-'),
        "count_dot": url.count('.'),
        "num_subdomains": 0 if not subdomain else subdomain.count('.') + 1,
        "has_https": 1 if parsed.scheme == 'https' else 0,
        "path_len": len(parsed.path or ""),
        "query_len": len(parsed.query or ""),
        "tld_len": len(suffix),
        "domain_len": len(domain),
        "suspicious_keyword": int(any(k in url.lower() for k in SCAM_KEYWORDS)),
    }
    return features


def prepare_features(df):
    X = df['url'].apply(lambda u: pd.Series(extract_features(u)))
    return X


def train_and_save(df, out_path='models/url_model.pkl'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if 'label' not in df.columns:
        raise ValueError('Dataset must have a `label` column with 0 (legit) or 1 (phish)')

    X = prepare_features(df)
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipeline, out_path)
    print(f"Saved trained model to {out_path}")


def main():
    print('Loading datasets...')
    df = load_datasets()
    print(f'Loaded {len(df)} rows')
    # Expecting columns: url,label
    if 'url' not in df.columns or 'label' not in df.columns:
        print('Dataset must contain `url` and `label` columns. Aborting.')
        return
    train_and_save(df)


if __name__ == '__main__':
    main()
