from flask import Flask, render_template, request, redirect
import re
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define safe domains
SAFE_DOMAINS = ['wikipedia.org', 'google.com', 'github.com', 'python.org', 'stackoverflow.com']

# Initialize models
def initialize_models():
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
            'http://click-now.win-prize.com'
        ],
        'label': [
            'benign', 'phishing', 'malware', 'phishing', 'benign',
            'benign', 'benign', 'benign', 'benign', 'phishing',
            'malware', 'spam'
        ]
    })
    url_vectorizer = TfidfVectorizer()
    url_X = url_vectorizer.fit_transform(url_data['url'])
    url_model = LogisticRegression()
    url_model.fit(url_X, url_data['label'])

    spam_data = pd.DataFrame({
        'message': [
            'Hello, how are you doing today?',
            'You won a FREE iPhone! Claim now!',
            'Your account has been suspended, login immediately!',
            'Reminder: project meeting at 2pm',
            'URGENT: Your invoice is overdue!',
            'Click here to claim your prize!',
            'Normal message without spam.',
            'Let’s catch up over lunch tomorrow',
            'Congratulations! You are a lucky winner!',
            'Call now to claim your cash reward!'
        ],
        'label': ['ham', 'spam', 'phishing', 'ham', 'phishing', 'spam', 'ham', 'ham', 'spam', 'spam']
    })
    spam_vectorizer = TfidfVectorizer()
    spam_X = spam_vectorizer.fit_transform(spam_data['message'])
    spam_model = LogisticRegression()
    spam_model.fit(spam_X, spam_data['label'])

    return url_model, url_vectorizer, spam_model, spam_vectorizer

url_model, url_vectorizer, spam_model, spam_vectorizer = initialize_models()

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
    
    return dict(
        get_icon_for_class=get_icon_for_class,
        get_risk_level=get_risk_level
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def url_detection(url):
    if any(domain in url for domain in SAFE_DOMAINS):
        return "benign"
    try:
        tfidf = url_vectorizer.transform([url])
        return url_model.predict(tfidf)[0]
    except:
        if any(x in url.lower() for x in ['login', 'bank', 'secure']):
            return "phishing"
        elif any(x in url.lower() for x in ['malware', 'trojan', 'virus']):
            return "malware"
        return "benign"

def message_detection(message):
    try:
        tfidf = spam_vectorizer.transform([message])
        prediction = spam_model.predict(tfidf)[0]
        confidence = max(spam_model.predict_proba(tfidf)[0]) * 100
        return prediction, confidence
    except:
        spam_indicators = ['win', 'free', 'click', 'urgent', 'cash', 'prize']
        score = sum(word in message.lower() for word in spam_indicators)
        if score > 2:
            return "spam", 90
        return "ham", 95

def analyze_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header.startswith(b'MZ') or header.startswith(b'\x7fELF'):
                return "malicious", "Executable file detected"
            if header.startswith(b'%PDF'):
                return "benign", "PDF file - no threats detected"
            try:
                content = header.decode('utf-8') + f.read(4096).decode('utf-8', errors='ignore')
                if re.search(r'(password|login|credit.?card)', content, re.IGNORECASE):
                    return "phishing", "Contains sensitive information patterns"
                return "benign", "No threats detected"
            except UnicodeDecodeError:
                return "unknown", "Binary file - cannot analyze content"
    except Exception as e:
        return "unknown", f"Analysis error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.form['url']
    if not url.startswith(('http://', 'https://')):
        return render_template('index.html', message="Invalid URL format", input_url=url)
    
    predicted_class = url_detection(url)

    return render_template('index.html',
        input_url=url,
        predicted_class=predicted_class,
        url_result={
            'classification': predicted_class,
            'message': "URL appears safe and no threats detected." if predicted_class == "benign" else "Threat detected in the URL."
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

@app.route('/scam/', methods=['POST'])
def scan_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", file_message="No file selected")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, analysis = analyze_file(filepath)

        try:
            os.remove(filepath)
        except:
            pass

        return render_template("index.html", 
            message=f"{result.upper()}: {analysis}",
            file_result={
                'filename': filename,
                'classification': result,
                'analysis': analysis
            })
    else:
        return render_template("index.html", file_message="Invalid file type. Only PDF/TXT allowed")

if __name__ == '__main__':
    app.run(debug=True)
