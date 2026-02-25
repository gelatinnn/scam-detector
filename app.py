from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
import time
import requests
import joblib
import pandas as pd
import train_model

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Try to load trained model pipeline if available
# Detect and load trained model pipeline from common filenames
model_pipeline = None
possible_model_paths = [
    os.path.join("models", "url_model.pkl"),
    os.path.join("models", "train.model.pkl"),
    os.path.join("models", "train_model.pkl"),
    os.path.join("models", "model.pkl"),
    "train.model.pkl",
    "train_model.pkl",
]
for p in possible_model_paths:
    if os.path.exists(p):
        try:
            model_pipeline = joblib.load(p)
            print(f"Loaded model from: {p}")
            break
        except Exception:
            model_pipeline = None

scam_keywords = [
    "gcash", "paymaya", "bank", "send money", "transfer",
    "win", "winner", "congratulations", "prize", "reward", "free", "promo",
    "urgent", "verify", "login", "confirm",
    "investment", "double your money", "easy profit", "job", "work from home", "shop now", "limited offer", "act fast"
]

suspicious_domains = [".xyz", ".tk", ".ml", ".cf", ".ga", ".shop", ".top"]

sensitive_requests = ["password", "otp", "pin", "card number", "account number"]
urgency_phrases = ["now", "today", "immediately", "within 24 hours"]

@app.route("/", methods=["GET", "POST"])
def index():
    # Try to surface any previous POST result stored in session (PRG pattern)
    result = session.pop("last_result", "") or ""
    reasons = session.pop("last_reasons", []) or []
    found_urls = session.pop("last_found_urls", []) or []

    if request.method == "POST":
        text = (request.form.get("input_text", "") or "").lower()

        # (Image upload/ OCR removed) - only processing pasted text/URLs

        # normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        reasons_set = set()

        # Keyword check (use word boundaries to avoid substrings)
        for word in scam_keywords:
            if re.search(r"\b" + re.escape(word) + r"\b", text):
                reasons_set.add(f"Suspicious keyword: {word}")

        # Domain check: detect suspicious TLDs in text
        domain_pattern = re.compile(r"\b[\w.-]+\.(?:xyz|tk|ml|cf|ga|shop|top)\b", re.IGNORECASE)
        for dom in domain_pattern.findall(text):
            reasons_set.add(f"Suspicious domain: {dom}")

        # Sensitive info request
        for s in sensitive_requests:
            if re.search(r"\b" + re.escape(s) + r"\b", text):
                reasons_set.add(f"Requests sensitive info: {s}")


        # URL patterns and weird URLs
        urls = re.findall(r'(https?://[^\s]+)', text)
        found_urls = []
        for url in urls:
            if len(url) > 60:
                reasons_set.add("Very long URL")
            if re.search(r"\d{3,}", url):
                reasons_set.add("URL has many numbers")

            # Optional VirusTotal reputation check (requires VIRUSTOTAL_API_KEY env var)
            vt_api_key = os.getenv("VIRUSTOTAL_API_KEY")
            if vt_api_key:
                try:
                    vt_result = check_virustotal_url(url, vt_api_key)
                    if vt_result is not None:
                        positives = vt_result.get("positives", 0)
                        if positives > 0:
                            reasons_set.add(f"VirusTotal positives: {positives}")
                        else:
                            # small positive signal if not flagged
                            reasons_set.add("VirusTotal: no engines flagged")
                except Exception:
                    # don't fail the whole check if VT lookup fails
                    pass

            # Model prediction (if trained model available)
            model_score = None
            if model_pipeline is not None:
                try:
                    feats = train_model.extract_features(url)
                    Xdf = pd.DataFrame([feats])
                    prob = model_pipeline.predict_proba(Xdf)[0][1]
                    model_score = float(prob)
                    reasons_set.add(f"Model score: {model_score:.2f}")
                except Exception:
                    model_score = None

            found_urls.append({"url": url, "model_score": model_score})

        # Final result: priority-based rather than simple count
        reason_count = len(reasons_set)
        # If VirusTotal flagged engines, treat as likely scam
        if any(r.startswith("VirusTotal positives:") and int(r.split(":",1)[1].strip()) > 0 for r in reasons_set):
            result = "Likely SCAM"
        # Sensitive info requests or suspicious domains are strong signals
        elif any(r.startswith("Requests sensitive info") or r.startswith("Suspicious domain") for r in reasons_set):
            result = "Possibly SCAM"
        # Suspicious keywords or URL anomalies -> at least possibly scam
        elif any(r.startswith("Suspicious keyword") or r in ("Very long URL", "URL has many numbers") for r in reasons_set):
            result = "Possibly SCAM"
        else:
            result = " Possibly SAFE"

        reasons = sorted(reasons_set)

        # Store the result in session and redirect to GET to avoid form re-submission on refresh
        session["last_result"] = result
        session["last_reasons"] = reasons
        session["last_found_urls"] = found_urls
        return redirect(url_for("index"))

    return render_template("index.html", result=result, reasons=reasons, found_urls=found_urls)


def check_virustotal_url(url, api_key, timeout=10):
    """Submit URL to VirusTotal and poll for analysis results.
    Returns a dict with at least 'positives' (int) or None on failure.
    """
    headers = {"x-apikey": api_key}
    try:
        post = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url}, timeout=timeout)
        post.raise_for_status()
        data = post.json()
        analysis_id = data.get("data", {}).get("id")
        if not analysis_id:
            return None

        # Poll analysis endpoint
        for _ in range(8):
            time.sleep(1)
            res = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers, timeout=timeout)
            if res.status_code != 200:
                continue
            js = res.json()
            attr = js.get("data", {}).get("attributes", {})
            if attr.get("status") == "completed":
                stats = attr.get("stats") or {}
                # stats usually contains keys like 'malicious' and 'suspicious'
                positives = stats.get("malicious", 0) + stats.get("suspicious", 0)
                return {"positives": positives, "stats": stats}
        return None
    except Exception:
        return None

if __name__ == "__main__":
    app.run(debug=True)