# personality_analyzer.py - COMPLETE STARTER CODE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🎭 TEXT TO PERSONALITY MAPPER - STARTING...")
print("=" * 50)

# Step 1: Check if dataset exists
print("1. Checking for dataset...")
if not os.path.exists('mbti_1.csv'):
    print("❌ ERROR: mbti_1.csv not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/datasnaek/mbti-type")
    print("And place it in the same folder as this script")
    exit()
else:
    print("✅ Dataset found!")

# Step 2: Load the data
print("\n2. Loading dataset...")
df = pd.read_csv('mbti_1.csv')
print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Step 3: Show sample data
print("\n3. Data sample:")
print(df.head())

# Step 4: Show MBTI type distribution
print("\n4. MBTI Type Distribution:")
type_counts = df['type'].value_counts()
print(type_counts)

# Step 5: Clean text data
print("\n5. Cleaning text data...")
def clean_text(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep letters and basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.!?,]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['cleaned_posts'] = df['posts'].apply(clean_text)
print("✅ Text cleaning completed!")

# Show cleaning example
print(f"\nSample before cleaning: {df['posts'].iloc[0][:100]}...")
print(f"Sample after cleaning: {df['cleaned_posts'].iloc[0][:100]}...")

# Step 6: Prepare for machine learning
print("\n6. Preparing machine learning model...")
X = df['cleaned_posts']
y = df['type']

# Use smaller sample for faster training (remove this line for full dataset)
df = df.sample(1000, random_state=42)  # Using 1000 samples for speed

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 7: Convert text to features
print("\n7. Creating text features...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Step 8: Train model
print("\n8. Training personality classifier...")
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train_tfidf, y_train)
print("✅ Model training completed!")

# Step 9: Check accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2%}")

# Step 10: Create prediction function
def predict_personality(text):
    """Predict personality from any text input"""
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(text_tfidf)[0]
    
    results = {}
    for i, personality_type in enumerate(model.classes_):
        results[personality_type] = round(probabilities[i] * 100, 2)
    
    # Sort by probability (highest first)
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Step 11: Test with sample texts
print("\n" + "=" * 50)
print("🧪 TESTING WITH SAMPLE TEXTS")
print("=" * 50)

test_samples = [
    "I love going to parties and meeting new people. Social events energize me and I enjoy being the center of attention!",
    "I prefer reading books alone at home. Deep conversations with close friends are better than large gatherings.",
    "I am very organized and make detailed plans. I believe rules and structure are important for success.",
    "I go with the flow and adapt to situations. I'm very empathetic and can understand how others feel easily."
]

for i, text in enumerate(test_samples, 1):
    print(f"\n--- Test {i} ---")
    print(f"Input: {text}")
    results = predict_personality(text)
    top_personality = list(results.keys())[0]
    top_confidence = list(results.values())[0]
    print(f"🏆 Predicted: {top_personality} ({top_confidence}% confidence)")
    print("Top 3 predictions:")
    for j, (p_type, prob) in enumerate(list(results.items())[:3]):
        print(f"  {j+1}. {p_type}: {prob}%")

# Step 12: Interactive Mode
print("\n" + "=" * 50)
print("🎮 INTERACTIVE MODE - TRY YOUR OWN TEXT!")
print("Type 'quit' to exit")
print("=" * 50)

while True:
    user_text = input("\n📝 Enter your text to analyze: ")
    
    if user_text.lower() == 'quit':
        print("👋 Thanks for using Personality Mapper!")
        break
        
    if len(user_text.strip()) < 10:
        print("⚠️  Please enter at least 10 characters")
        continue
        
    results = predict_personality(user_text)
    top_personality = list(results.keys())[0]
    top_confidence = list(results.values())[0]
    
    print(f"\n🔮 YOUR PERSONALITY ANALYSIS:")
    print(f"Most likely: {top_personality} ({top_confidence}% confidence)")
    print("\nAll probabilities:")
    for p_type, prob in results.items():
        print(f"  {p_type}: {prob}%")
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    top_5 = dict(list(results.items())[:5])
    plt.bar(top_5.keys(), top_5.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    plt.title(f'Your Personality Prediction\n(Top match: {top_personality} - {top_confidence}%)')
    plt.ylabel('Probability (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()