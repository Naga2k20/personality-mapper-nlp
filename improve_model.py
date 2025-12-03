# improve_model.py - Advanced techniques
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Try different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Test each model and compare accuracy