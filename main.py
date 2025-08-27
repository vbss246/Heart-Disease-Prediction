from preprocessing import load_and_preprocess_data
from models import get_models
from stacking_pipeline import StackingClassifierPipeline
from explainability import explain_model

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Data/heart_2020_cleaned.csv')

# Initialize models
models = get_models()

# Build and train stacking ensemble
stacker = StackingClassifierPipeline(models)
stacker.fit(X_train, y_train)
preds = stacker.predict(X_test)

# Evaluate performance
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(classification_report(y_test, preds))

# Explain the ensemble predictions
explain_model(stacker, X_test)
