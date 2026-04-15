# Lightweight Phishing Website Detection System

## Live Demo
https://lightweight-phishing-website-detector.onrender.com


## Overview

This project implements a lightweight phishing detection system that analyzes URLs and web pages to determine whether a website is legitimate or potentially malicious.

The main objective is to provide a fast and practical solution that works even when full webpage data is not available. The system combines machine learning models with rule-based risk analysis to produce interpretable results.

The application provides:
- Prediction (Legitimate / Phishing)
- Probability score
- Risk level (Low / Medium / High)
- Explanation of detected signals


## Approach

The system uses a hybrid approach combining two machine learning models along with domain-based risk scoring.

### Web-based Model

This model extracts features from the webpage when it is accessible. It focuses on structural and behavioral signals such as:
- Presence of forms and password fields
- iFrames and popups
- External form submissions
- Page metadata and responsiveness

This model generally provides better accuracy but depends on successful webpage retrieval.


### URL-based Model

This model works only with the URL string and does not require webpage access. It extracts features such as:
- URL length and entropy
- Number of digits, symbols, and special characters
- Suspicious keywords (login, verify, secure, update)
- Domain structure and subdomain patterns

This makes the system lightweight and reliable even when webpage data is unavailable.


## Prediction Modes

The application supports three prediction modes:

- **Auto (default)**  
  Attempts to use the web-based model first. If the webpage cannot be accessed, it automatically falls back to the URL-based model.

- **Web Model**  
  Uses only webpage-based features.

- **URL Model**  
  Uses only URL-based features.


## Risk Scoring

In addition to model predictions, a rule-based risk score is calculated using domain-related signals such as:
- Suspicious keywords in the URL
- Brand impersonation patterns
- Hyphen usage and multiple subdomains
- Suspicious top-level domains (TLDs)

This score is combined with model output to improve interpretability and provide clearer explanations.


## Project Structure

```

phishing-app/
│
├── app.py                  # Flask backend and prediction logic
├── requirements.txt        # Python dependencies
│
├── models/                 # Trained machine learning models
│   ├── best_url_model.pkl
│   ├── best_web_model.pkl
│   ├── url_model_columns.pkl
│   └── web_model_columns.pkl
├── notebooks/
│   └── phishing_model_training.ipynb 
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   └── style.css           # Styling (if separated)

````


## How It Works

1. The user enters a URL in the web interface  
2. The selected prediction mode is sent to the backend  
3. Features are extracted from:
   - The webpage (if accessible)
   - The URL itself  
4. The selected model generates a phishing probability  
5. A domain-based risk score is calculated  
6. The final result is displayed with explanation  



## Running the Project Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
````

2. Run the application:

```bash
python app.py
```

3. Open in browser:

```
http://127.0.0.1:5000
```

## Model Development

The training and experimentation code is available in:

notebooks/phishing_model_training.ipynb


## Deployment

The application can be deployed using services like Render.

Start command:

```bash
gunicorn app:app
```


## Notes

* The system is designed to be lightweight and efficient
* Web-based analysis improves accuracy but is not always available
* URL-based analysis ensures fallback support


## Disclaimer

This tool provides an automated assessment based on machine learning models and heuristic rules. The results are probabilistic and should be used as a supporting tool, not as a final security decision.
