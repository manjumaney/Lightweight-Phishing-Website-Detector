import os
import re
import joblib
import pandas as pd
import requests

from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from difflib import SequenceMatcher

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

web_model = joblib.load(os.path.join(BASE_DIR, "models", "best_web_model.pkl"))
web_columns = joblib.load(os.path.join(BASE_DIR, "models", "web_model_columns.pkl"))

url_model = joblib.load(os.path.join(BASE_DIR, "models", "best_url_model.pkl"))
url_columns = joblib.load(os.path.join(BASE_DIR, "models", "url_model_columns.pkl"))


def safe_ratio(a, b):
    return a / b if b not in [0, None] else 0.0


def count_digits(s):
    return sum(ch.isdigit() for ch in s)


def count_letters(s):
    return sum(ch.isalpha() for ch in s)


def count_special_chars(s):
    return sum(not ch.isalnum() for ch in s)


def is_ip_address(hostname):
    if not hostname:
        return 0
    pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return int(bool(re.match(pattern, hostname)))


def get_tld(hostname):
    if not hostname or "." not in hostname:
        return ""
    return hostname.split(".")[-1].lower()


def count_subdomains(hostname):
    if not hostname:
        return 0
    parts = hostname.split(".")
    return max(len(parts) - 2, 0)


def has_obfuscation(url):
    return int("%" in url or "@" in url)


def count_obfuscated_chars(url):
    return url.count("%") + url.count("@")


def count_redirects(response):
    try:
        return len(response.history)
    except Exception:
        return 0


def format_probability(prob):
    percentage = float(prob) * 100
    if percentage < 0.01:
        return "<0.01%"
    return f"{percentage:.2f}%"


def get_risk_level(prob):
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


KNOWN_BRANDS = [
    "trezor", "paypal", "coinbase", "binance", "metamask",
    "ledger", "bankofamerica", "chase", "wellsfargo",
    "amazon", "microsoft", "apple", "google", "facebook"
]

SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "update", "wallet",
    "account", "auth", "signin", "recover", "support"
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "shop", "live", "buzz", "rest",
    "fit", "country", "stream", "gq", "pro"
}


def get_hostname_parts(url):
    parsed = urlparse(url)
    hostname = parsed.netloc.lower().split(":")[0]
    parts = hostname.split(".") if hostname else []
    return hostname, parts


def get_registered_domain_like(hostname_parts):
    if len(hostname_parts) >= 2:
        return ".".join(hostname_parts[-2:])
    if len(hostname_parts) == 1:
        return hostname_parts[0]
    return ""


def get_domain_without_tld(domain_like):
    if "." in domain_like:
        return domain_like.rsplit(".", 1)[0]
    return domain_like


def get_tld_from_domain(domain_like):
    if "." in domain_like:
        return domain_like.rsplit(".", 1)[-1]
    return ""


def similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_present_brands(text, brands):
    text = text.lower()
    return [b for b in brands if b in text]


def typo_similarity_features(domain_text, brands, threshold=0.75):
    max_sim = 0.0
    closest_brand = None

    for brand in brands:
        sim = similarity_score(domain_text, brand)
        if sim > max_sim:
            max_sim = sim
            closest_brand = brand

    return {
        "MaxBrandSimilarity": max_sim,
        "HasTypoBrandSimilarity": int(max_sim >= threshold and closest_brand not in domain_text),
        "ClosestBrand": closest_brand
    }


def extract_domain_risk_features(url):
    hostname, parts = get_hostname_parts(url)
    registered_domain = get_registered_domain_like(parts)
    domain_no_tld = get_domain_without_tld(registered_domain)
    tld = get_tld_from_domain(registered_domain)

    full_text = url.lower()
    host_text = hostname.lower()
    domain_text = domain_no_tld.lower()

    present_brands = find_present_brands(full_text, KNOWN_BRANDS)
    suspicious_words_present = [w for w in SUSPICIOUS_WORDS if w in full_text]

    typo_info = typo_similarity_features(domain_text.replace("-", ""), KNOWN_BRANDS, threshold=0.75)

    features = {
        "HasKnownBrand": int(len(present_brands) > 0),
        "BrandCountInURL": len(present_brands),
        "HasSuspiciousWord": int(len(suspicious_words_present) > 0),
        "SuspiciousWordCount": len(suspicious_words_present),
        "HasBrandAndSuspiciousWord": int(len(present_brands) > 0 and len(suspicious_words_present) > 0),
        "HyphenCountInHost": host_text.count("-"),
        "HasHyphenInHost": int("-" in host_text),
        "HasManySubdomains": int(max(len(parts) - 2, 0) >= 2),
        "HasSuspiciousTLD": int(tld in SUSPICIOUS_TLDS),
        "HasTypoBrandSimilarity": typo_info["HasTypoBrandSimilarity"],
    }

    for brand in KNOWN_BRANDS:
        features[f"Brand_{brand}"] = int(brand in full_text)

    return features


def compute_url_risk_score(url):
    f = extract_domain_risk_features(url)

    score = 0
    if f["HasSuspiciousWord"]:
        score += 15
    if f["HasBrandAndSuspiciousWord"]:
        score += 25
    if f["HasHyphenInHost"]:
        score += 10
    if f["HasManySubdomains"]:
        score += 10
    if f["HasSuspiciousTLD"]:
        score += 10
    if f["HasTypoBrandSimilarity"]:
        score += 20

    return min(score, 100), f


def explain_url_risk(url):
    features = extract_domain_risk_features(url)
    reasons = []

    if features["HasBrandAndSuspiciousWord"]:
        reasons.append("brand name combined with suspicious keyword")
    elif features["HasSuspiciousWord"]:
        reasons.append("contains suspicious keyword in URL")

    if features["HasHyphenInHost"]:
        reasons.append("hyphenated domain structure")

    if features["HasManySubdomains"]:
        reasons.append("many subdomains present")

    if features["HasSuspiciousTLD"]:
        reasons.append("suspicious top-level domain")

    if features["HasTypoBrandSimilarity"]:
        reasons.append("possible brand impersonation or typo similarity")

    present_brands = [
        k.replace("Brand_", "") for k, v in features.items()
        if k.startswith("Brand_") and v == 1
    ]
    if present_brands:
        reasons.append(f"brand keyword detected: {', '.join(present_brands)}")

    if not reasons:
        reasons.append("no major URL risk pattern detected")

    return reasons


def extract_features_from_url(url, timeout=10):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    full_url = url.strip()

    feature_dict = {
        "URLLength": len(full_url),
        "DomainLength": len(domain),
        "IsDomainIP": is_ip_address(domain),
        "TLDLength": len(get_tld(domain)),
        "NoOfSubDomain": count_subdomains(domain),
        "HasObfuscation": has_obfuscation(full_url),
        "NoOfObfuscatedChar": count_obfuscated_chars(full_url),
        "ObfuscationRatio": safe_ratio(count_obfuscated_chars(full_url), len(full_url)),
        "NoOfLettersInURL": count_letters(full_url),
        "LetterRatioInURL": safe_ratio(count_letters(full_url), len(full_url)),
        "NoOfDegitsInURL": count_digits(full_url),
        "DegitRatioInURL": safe_ratio(count_digits(full_url), len(full_url)),
        "NoOfEqualsInURL": full_url.count("="),
        "NoOfQMarkInURL": full_url.count("?"),
        "NoOfAmpersandInURL": full_url.count("&"),
        "NoOfOtherSpecialCharsInURL": len(re.findall(r"[^A-Za-z0-9=?&:/._\-]", full_url)),
        "SpacialCharRatioInURL": safe_ratio(count_special_chars(full_url), len(full_url)),
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
        "Robots": 0,
        "IsResponsive": 0,
        "NoOfURLRedirect": 0,
        "NoOfSelfRedirect": 0,
        "HasDescription": 0,
        "NoOfPopup": 0,
        "NoOfiFrame": 0,
        "HasExternalFormSubmit": 0,
        "HasSocialNet": 0,
        "HasSubmitButton": 0,
        "HasHiddenFields": 0,
        "HasPasswordField": 0,
        "Bank": 0,
        "Pay": 0,
        "Crypto": 0,
    }

    fetch_success = 0
    fetch_error = None

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()

        html = response.text
        content_type = response.headers.get("Content-Type", "").lower()

        if "xml" in content_type and "html" not in content_type:
            soup = BeautifulSoup(html, "xml")
        else:
            soup = BeautifulSoup(html, "html.parser")

        fetch_success = 1
        feature_dict["NoOfURLRedirect"] = count_redirects(response)

        robots_meta = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
        feature_dict["Robots"] = int(robots_meta is not None)

        viewport_meta = soup.find("meta", attrs={"name": re.compile("^viewport$", re.I)})
        feature_dict["IsResponsive"] = int(viewport_meta is not None)

        desc_meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        feature_dict["HasDescription"] = int(desc_meta is not None)

        html_lower = html.lower()
        feature_dict["NoOfPopup"] = html_lower.count("window.open")
        feature_dict["NoOfiFrame"] = len(soup.find_all("iframe"))

        forms = soup.find_all("form")
        external_form_submit = 0
        has_submit = 0
        hidden_fields = 0
        password_fields = 0

        for form in forms:
            action = form.get("action", "")
            if action and action.startswith("http"):
                action_domain = urlparse(action).netloc.lower()
                if action_domain and action_domain != domain:
                    external_form_submit = 1

            if form.find("input", attrs={"type": re.compile("^submit$", re.I)}):
                has_submit = 1

            hidden_fields += len(form.find_all("input", attrs={"type": re.compile("^hidden$", re.I)}))
            password_fields += len(form.find_all("input", attrs={"type": re.compile("^password$", re.I)}))

        feature_dict["HasExternalFormSubmit"] = int(external_form_submit)
        feature_dict["HasSubmitButton"] = int(has_submit)
        feature_dict["HasHiddenFields"] = int(hidden_fields > 0)
        feature_dict["HasPasswordField"] = int(password_fields > 0)

        page_text = soup.get_text(" ", strip=True).lower()
        feature_dict["Bank"] = int(any(word in page_text for word in ["bank", "banking"]))
        feature_dict["Pay"] = int(any(word in page_text for word in ["pay", "payment", "billing"]))
        feature_dict["Crypto"] = int(any(word in page_text for word in ["crypto", "bitcoin", "wallet"]))

    except requests.exceptions.RequestException as e:
        fetch_error = str(e)

    return feature_dict, fetch_success, fetch_error


def build_feature_dataframe(url, web_columns):
    features, fetch_success, fetch_error = extract_features_from_url(url)
    row = pd.DataFrame([features])

    for col in web_columns:
        if col not in row.columns:
            row[col] = 0

    extra_cols = [c for c in row.columns if c not in web_columns]
    if extra_cols:
        row = row.drop(columns=extra_cols)

    row = row[web_columns]
    return row, fetch_success, fetch_error


def build_url_model_features(url, url_columns):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url.strip()

    features = {
        "url_has_login": int("login" in full_url.lower()),
        "url_has_client": int("client" in full_url.lower()),
        "url_has_server": int("server" in full_url.lower()),
        "url_has_admin": int("admin" in full_url.lower()),
        "url_has_ip": is_ip_address(domain),
        "url_isshorted": int(any(short in domain for short in ["bit.ly", "tinyurl.com", "goo.gl", "t.co"])),
        "url_len": len(full_url),
        "url_entropy": safe_ratio(len(set(full_url)), len(full_url)),
        "url_hamming_1": full_url.count("1") / max(len(full_url), 1),
        "url_hamming_00": full_url.count("0") / max(len(full_url), 1),
        "url_2bentropy": safe_ratio(len(set(full_url[:max(2, len(full_url) // 2)])), max(2, len(full_url) // 2)),
        "url_count_dot": full_url.count("."),
        "url_count_https": full_url.lower().count("https"),
        "url_count_http": full_url.lower().count("http"),
        "url_count_perc": full_url.count("%"),
        "url_count_hyphen": full_url.count("-"),
        "url_count_www": full_url.lower().count("www"),
        "url_count_atrate": full_url.count("@"),
        "url_count_hash": full_url.count("#"),
        "url_count_semicolon": full_url.count(";"),
        "url_count_underscore": full_url.count("_"),
        "url_count_ques": full_url.count("?"),
        "url_count_equal": full_url.count("="),
        "url_count_amp": full_url.count("&"),
        "url_count_letter": count_letters(full_url),
        "url_count_digit": count_digits(full_url),
        "url_count_sensitive_financial_words": sum(
            word in full_url.lower() for word in ["bank", "pay", "wallet", "crypto"]
        ),
        "url_count_sensitive_words": sum(
            word in full_url.lower() for word in ["login", "secure", "verify", "update", "account"]
        ),
        "url_nunique_chars_ratio": safe_ratio(len(set(full_url)), len(full_url)),
        "path_len": len(path),
        "path_count_no_of_dir": path.count("/"),
        "path_count_no_of_embed": path.count("//"),
        "path_count_zero": path.count("0"),
        "path_count_pertwent": path.count("%20"),
        "path_has_any_sensitive_words": int(
            any(word in path.lower() for word in ["login", "secure", "verify", "account"])
        ),
        "path_count_lower": sum(ch.islower() for ch in path),
        "path_count_upper": sum(ch.isupper() for ch in path),
        "path_count_nonascii": sum(ord(ch) > 127 for ch in path),
        "path_has_singlechardir": int(any(len(part) == 1 for part in path.split("/") if part)),
        "path_has_upperdir": int(any(part.isupper() for part in path.split("/") if part)),
        "query_len": len(query),
        "query_count_components": len(query.split("&")) if query else 0,
        "pdomain_len": len(domain),
        "pdomain_count_hyphen": domain.count("-"),
        "pdomain_count_atrate": domain.count("@"),
        "pdomain_count_non_alphanum": sum(not ch.isalnum() and ch != "." for ch in domain),
        "pdomain_count_digit": count_digits(domain),
        "tld_len": len(get_tld(domain)),
        "tld_is_sus": int(get_tld(domain) in {"xyz", "top", "click", "shop", "live", "buzz", "gq", "pro"}),
        "pdomain_min_distance": 0,
        "subdomain_len": len(domain.split(".")[0]) if "." in domain else len(domain),
        "subdomain_count_dot": max(domain.count(".") - 1, 0),
    }

    row = pd.DataFrame([features])

    for col in url_columns:
        if col not in row.columns:
            row[col] = 0

    extra_cols = [c for c in row.columns if c not in url_columns]
    if extra_cols:
        row = row.drop(columns=extra_cols)

    return row[url_columns]


def make_result(url, mode_text, final_prob, risk_score, reasons, error=None, threshold=0.3):
    pred = int(final_prob >= threshold)
    label = "Phishing" if pred == 1 else "Legitimate"

    return {
        "url": url,
        "mode": mode_text,
        "label": label,
        "prediction": pred,
        "final_probability": format_probability(final_prob),
        "url_risk_score": f"{risk_score}/100",
        "url_risk_level": get_risk_level(final_prob),
        "url_risk_reasons": "; ".join(reasons),
        "error": error
    }


def predict_live_final(url, web_model, web_columns, url_model, url_columns, threshold=0.3, selected_model="auto"):
    url_risk_score, _ = compute_url_risk_score(url)
    url_risk_prob = url_risk_score / 100.0
    url_risk_reasons = explain_url_risk(url)

    if selected_model == "web":
        X_web_live, fetch_success, fetch_error = build_feature_dataframe(url, web_columns)

        if fetch_success != 1:
            return {
                "url": url,
                "mode": "Web model",
                "label": "Unavailable",
                "prediction": 0,
                "final_probability": "0.00%",
                "url_risk_score": f"{url_risk_score}/100",
                "url_risk_level": get_risk_level(url_risk_prob),
                "url_risk_reasons": "; ".join(url_risk_reasons),
                "error": fetch_error or "Webpage could not be fetched for Web model prediction."
            }

        web_prob = web_model.predict_proba(X_web_live)[0, 1]
        final_prob = 0.85 * web_prob + 0.15 * url_risk_prob

        return make_result(
            url=url,
            mode_text="Web model",
            final_prob=final_prob,
            risk_score=url_risk_score,
            reasons=url_risk_reasons,
            error=None,
            threshold=threshold
        )

    if selected_model == "url":
        X_url_live = build_url_model_features(url, url_columns)
        url_model_prob = url_model.predict_proba(X_url_live)[0, 1]
        final_prob = 0.75 * url_model_prob + 0.25 * url_risk_prob

        return make_result(
            url=url,
            mode_text="URL model",
            final_prob=final_prob,
            risk_score=url_risk_score,
            reasons=url_risk_reasons,
            error=None,
            threshold=threshold
        )

    X_web_live, fetch_success, fetch_error = build_feature_dataframe(url, web_columns)

    if fetch_success == 1:
        web_prob = web_model.predict_proba(X_web_live)[0, 1]
        final_prob = 0.85 * web_prob + 0.15 * url_risk_prob

        return make_result(
            url=url,
            mode_text="Auto → Web model",
            final_prob=final_prob,
            risk_score=url_risk_score,
            reasons=url_risk_reasons,
            error=None,
            threshold=threshold
        )

    X_url_live = build_url_model_features(url, url_columns)
    url_model_prob = url_model.predict_proba(X_url_live)[0, 1]
    final_prob = 0.75 * url_model_prob + 0.25 * url_risk_prob

    return make_result(
        url=url,
        mode_text="Auto → URL model (fallback)",
        final_prob=final_prob,
        risk_score=url_risk_score,
        reasons=url_risk_reasons,
        error=fetch_error,
        threshold=threshold
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    selected_model = data.get("model", "auto").strip().lower()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Please enter a full URL starting with http:// or https://"}), 400

    if selected_model not in {"auto", "web", "url"}:
        selected_model = "auto"

    result = predict_live_final(
        url,
        web_model,
        web_columns,
        url_model,
        url_columns,
        selected_model=selected_model
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)