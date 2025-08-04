from flask import Flask, render_template, request
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Define safe domains
SAFE_DOMAINS = ['wikipedia.org', 'google.com', 'github.com', 'python.org', 'stackoverflow.com']
# Initialize models with improved training data
def initialize_models():
    # Enhanced URL dataset
    url_data = pd.DataFrame({
        'url': [
            'http://example.com',
            'http://phishing.com/login',
            'http://malware.com/download',
            'https://secure.bank.com',
            'https://www.wikipedia.org',
            'https://github.com',
            'https://www.google.com',
            'https://docs.python.org',
            'https://stackoverflow.com',
            'http://scam-login.biz',
            'http://trojan-malware.site',
            'http://click-now.win-prize.com',
            'https://paypal.verify-account.com',
            'https://amazon.account-security.com',
            'https://netflix.sign-in-now.com',
            'https://appleid.apple.com.verify.account.com',
            'https://microsoft.account-security.com',
            'https://linkedin.verify-account.com',
            'https://twitter.account-security.com',
            'https://facebook.verify-login.com',
            'https://instagram.account-confirmation.com',
            'https://dropbox.verify-account.com',
            'https://ebay.account-security.com',
            'https://wellsfargo.secure-login.com',
            'https://chase.verify-account.com',
            'https://bankofamerica.secure-login.com',
            'https://citibank.verify-account.com',
            'https://steam.community.login.secure.com',
            'https://epicgames.account.verify.com',
            'https://spotify.account-security.com'
        ],
        'label': [
            'benign', 'phishing', 'malware', 'phishing', 'benign',
            'benign', 'benign', 'benign', 'benign', 'phishing',
            'malware', 'spam', 'phishing', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing'
        ]
    })
    
    # Add more benign examples using pd.concat instead of append
    additional_urls = []
    for i in range(20):
        additional_urls.append({
            'url': f'https://legitimate-site-{i}.com/page/content/article',
            'label': 'benign'
        })
    
    url_data = pd.concat([url_data, pd.DataFrame(additional_urls)], ignore_index=True)
    
    # Extract features
    url_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', max_features=5000)
    url_X = url_vectorizer.fit_transform(url_data['url'])
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(url_X, url_data['label'], test_size=0.2, random_state=42)
    
    url_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    url_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = url_model.predict(X_test)
    url_accuracy = accuracy_score(y_test, y_pred)
    print(f"URL Model Accuracy: {url_accuracy:.2f}")

    # Enhanced spam dataset
    spam_data = pd.DataFrame({
        'message': [
            'Hello, how are you doing today?',
            'You won a FREE iPhone! Claim now!',
            'Your account has been suspended, login immediately!',
            'Reminder: project meeting at 2pm',
            'URGENT: Your invoice is overdue!',
            'Click here to claim your prize!',
            'Normal message without spam.',
            'Let\'s catch up over lunch tomorrow',
            'Congratulations! You are a lucky winner!',
            'Call now to claim your cash reward!',
            'Your package has been delivered',
            'Your Amazon order #12345 has shipped',
            'Your Netflix subscription is due for renewal',
            'Your bank account has been compromised',
            'Your PayPal account needs verification',
            'Your Microsoft account has been locked',
            'Your Facebook account has been hacked',
            'Your Instagram account has been suspended',
            'Your Twitter account has been compromised',
            'Your LinkedIn account has been locked',
            'Your Dropbox account has been suspended',
            'Your eBay account has been compromised',
            'Your Wells Fargo account has been locked',
            'Your Chase account has been suspended',
            'Your Bank of America account has been compromised',
            'Your Citibank account has been locked',
            'Your Steam account has been suspended',
            'Your Epic Games account has been compromised',
            'Your Spotify account has been locked'
        ],
        'label': [
            'ham', 'spam', 'phishing', 'ham', 'phishing', 
            'spam', 'ham', 'ham', 'spam', 'spam',
            'ham', 'ham', 'ham', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing', 'phishing',
            'phishing', 'phishing', 'phishing', 'phishing'
        ]
    })
    
    # Add more ham examples using pd.concat
    additional_messages = []
    for i in range(20):
        additional_messages.append({
            'message': f'This is a normal conversation message number {i} about regular topics',
            'label': 'ham'
        })
    
    spam_data = pd.concat([spam_data, pd.DataFrame(additional_messages)], ignore_index=True)
    
    # Extract features
    spam_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words='english')
    spam_X = spam_vectorizer.fit_transform(spam_data['message'])
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(spam_X, spam_data['label'], test_size=0.2, random_state=42)
    
    spam_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    spam_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = spam_model.predict(X_test)
    spam_accuracy = accuracy_score(y_test, y_pred)
    print(f"Spam Model Accuracy: {spam_accuracy:.2f}")

    return url_model, url_vectorizer, spam_model, spam_vectorizer, url_accuracy, spam_accuracy

url_model, url_vectorizer, spam_model, spam_vectorizer, url_accuracy, spam_accuracy = initialize_models()

@app.context_processor
def utility_processor():
    def get_icon_for_class(className):
        icons = {
            'benign': 'check-circle',
            'ham': 'comment-check',
            'defacement': 'exclamation-triangle',
            'malware': 'virus',
            'phishing': 'fish',
            'spam': 'comment-slash',
            'malicious': 'skull-crossbones'
        }
        return icons.get(className.lower(), 'question-circle')

    def get_risk_level(className):
        levels = {
            'benign': 'Low',
            'ham': 'Low',
            'defacement': 'Medium',
            'spam': 'Medium',
            'malware': 'High',
            'phishing': 'Critical',
            'malicious': 'Critical'
        }
        return levels.get(className.lower(), 'Unknown')
    
    def get_model_accuracies():
        return {
            'url_accuracy': f"{url_accuracy * 100:.1f}%",
            'spam_accuracy': f"{spam_accuracy * 100:.1f}%"
        }
    
    return dict(
        get_icon_for_class=get_icon_for_class,
        get_risk_level=get_risk_level,
        get_model_accuracies=get_model_accuracies
    )

def url_detection(url):
    if any(domain in url for domain in SAFE_DOMAINS):
        return "benign", 99.9
    
    try:
        tfidf = url_vectorizer.transform([url])
        prediction = url_model.predict(tfidf)[0]
        confidence = max(url_model.predict_proba(tfidf)[0]) * 100
        return prediction, confidence
    except Exception as e:
        print(f"URL detection error: {str(e)}")
        if any(x in url.lower() for x in ['login', 'bank', 'secure', 'verify', 'account']):
            return "phishing", 85.0
        elif any(x in url.lower() for x in ['malware', 'trojan', 'virus', 'download']):
            return "malware", 90.0
        return "benign", 95.0

def message_detection(message):
    try:
        tfidf = spam_vectorizer.transform([message])
        prediction = spam_model.predict(tfidf)[0]
        confidence = max(spam_model.predict_proba(tfidf)[0]) * 100
        
        # Additional heuristic checks
        spam_indicators = ['win', 'free', 'click', 'urgent', 'cash', 'prize', 'verify', 'account', 'suspended', 'locked']
        score = sum(word in message.lower() for word in spam_indicators)
        
        if prediction == 'ham' and score > 2:
            return "spam", max(confidence, 80.0)
        elif prediction != 'ham' and score > 0:
            confidence = min(100, confidence + (score * 5))
            
        return prediction, confidence
    except Exception as e:
        print(f"Message detection error: {str(e)}")
        spam_indicators = ['win', 'free', 'click', 'urgent', 'cash', 'prize']
        score = sum(word in message.lower() for word in spam_indicators)
        if score > 2:
            return "spam", 90.0
        return "ham", 95.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.form['url']
    if not url.startswith(('http://', 'https://')):
        return render_template('index.html', message="Invalid URL format", input_url=url)
    
    predicted_class, confidence = url_detection(url)

    return render_template('index.html',
        input_url=url,
        predicted_class=predicted_class,
        url_result={
            'classification': predicted_class,
            'confidence': f"{confidence:.1f}",
            'message': "URL appears safe and no threats detected." if predicted_class == "benign" else "Threat detected in the URL.",
            'is_threat': predicted_class != "benign"
        }
    )

@app.route('/detect-spam', methods=['POST'])
def detect_spam():
    message = request.form['message']
    predicted_class, confidence = message_detection(message)
    return render_template("index.html", 
        spam_result={
            'is_spam': predicted_class in ['spam', 'phishing'],
            'classification': predicted_class,
            'confidence': f"{confidence:.1f}",
            'analysis': "High spam probability" if predicted_class != 'ham' else "Legitimate message"
        })

if __name__ == '__main__':
    app.run(debug=True)