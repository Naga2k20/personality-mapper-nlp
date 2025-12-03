# advanced_model.py - Better accuracy with ensemble methods
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

print("🚀 ADVANCED PERSONALITY MODEL TRAINING...")

# Load and clean data
df = pd.read_csv('mbti_1.csv')

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\.!?,]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

df['cleaned_posts'] = df['posts'].apply(clean_text)

# Prepare data
X = df['cleaned_posts']
y = df['type']

# Use more data for better accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Enhanced TF-IDF with more features
vectorizer = TfidfVectorizer(
    max_features=3000, 
    stop_words='english',
    ngram_range=(1, 2),  # Use single words and word pairs
    min_df=2,            # Ignore very rare words
    max_df=0.8           # Ignore very common words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Try multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20),
    'SVM': SVC(probability=True, random_state=42, C=1.0, kernel='linear')
}

# Train and evaluate each model
best_accuracy = 0
best_model = None
best_model_name = ""

print("\n🧪 TRAINING MULTIPLE MODELS...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {accuracy:.2%}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n🏆 BEST MODEL: {best_model_name} with {best_accuracy:.2%} accuracy")

# Create ensemble model (combination of all models)
print("\n🔗 CREATING ENSEMBLE MODEL...")
ensemble = VotingClassifier(
    estimators=[('lr', models['Logistic Regression']), 
                ('rf', models['Random Forest']), 
                ('svm', models['SVM'])],
    voting='soft'
)

ensemble.fit(X_train_tfidf, y_train)
ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test_tfidf))
print(f"🎯 ENSEMBLE MODEL Accuracy: {ensemble_accuracy:.2%}")

# Save the best model
if ensemble_accuracy >= best_accuracy:
    final_model = ensemble
    final_accuracy = ensemble_accuracy
else:
    final_model = best_model
    final_accuracy = best_accuracy

# Save model and vectorizer
model_data = {
    'model': final_model,
    'vectorizer': vectorizer,
    'accuracy': final_accuracy
}

joblib.dump(model_data, 'advanced_personality_model.pkl')
print(f"💾 MODEL SAVED with {final_accuracy:.2%} accuracy!")

# Test with sample texts
def predict_personality_advanced(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    probabilities = final_model.predict_proba(text_tfidf)[0]
    
    results = {}
    for i, personality_type in enumerate(final_model.classes_):
        results[personality_type] = round(probabilities[i] * 100, 2)
    
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Test the improved model
test_texts = [
    "I love socializing and meeting new people at parties!",
    "I enjoy analyzing complex problems and reading scientific journals.",
    "I'm very organized and always make detailed plans for everything.",
    "I'm creative and emotional, I love painting and writing poetry."
]

print("\n" + "="*50)
print("🧪 TESTING IMPROVED MODEL")
print("="*50)

for i, text in enumerate(test_texts, 1):
    results = predict_personality_advanced(text)
    top_personality = list(results.keys())[0]
    top_confidence = list(results.values())[0]
    print(f"Test {i}: '{text}'")
    print(f"🏆 Prediction: {top_personality} ({top_confidence}% confidence)")
    print("---")