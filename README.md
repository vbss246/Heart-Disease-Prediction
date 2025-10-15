# Meta-Ensemble Learning for Heart Disease Prediction
### A Stacking-Based Approach with Explainable AI

This repository implements a robust heart disease prediction system using 
meta-ensemble machine learning (stacking) combined with explainable AI (XAI) techniques. 
The pipeline covers data preprocessing, model_training, evaluation, and interpretation, 
providing both accuracy and transparency.

---

## Repository Structure

| File/Folder              | Description |
|---------------------------|-------------|
| README.md                | Project documentation and usage guide |
| heart_2020_cleaned.csv.zip | Cleaned heart disease dataset (compressed CSV format) |
| preprocessing.py          | Data loading, cleaning, and feature engineering routines |
| models.py                 | Base learners and meta-estimator definitions for stacking |
| stacking_pipeline.py      | Stacking ensemble implementation and orchestration |
| main.py                   | Entry point for running the full workflow |
| explainability.py         | Model interpretation using SHAP/LIME |

---

## Key Features

- **Meta-Ensemble Learning (Stacking):** Combines multiple predictive models for better generalization.  
- **Explainable AI (XAI):** Provides model transparency using SHAP/LIME to highlight feature importance and decision reasoning.  
- **Modular Design:** Independent scripts for preprocessing, modeling, stacking, and interpretation.  
- **Flexible Dataset Integration:** Uses the CDC’s 2020 cleaned heart disease dataset for structured training and evaluation.  

---

## Getting Started

### Prerequisites
- Python 3.8+  
- Dependencies listed in `requirements.txt` (scikit-learn, pandas, numpy, matplotlib, shap, lime, etc.)  

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/vbss246/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Install dependencies
pip install -r requirements.txt

# Unzip dataset
unzip heart_2020_cleaned.csv.zip
```

### Run Workflow
```bash
python main.py
```

---

## Usage Workflow

- `preprocessing.py` → Preprocess and clean the dataset  
- `models.py` → Define base learners and meta-model  
- `stacking_pipeline.py` → Train and evaluate stacking ensemble  
- `explainability.py` → Generate interpretability reports (SHAP, LIME, feature importance plots)  

---

## Results & Performance

- Achieves high prediction accuracy via stacking multiple models.  
- Provides explainability reports to ensure medical and clinical interpretability.  
- Demonstrates which health features (e.g., age, cholesterol, BMI, smoking) drive predictions.  

---

## References

- Dataset: CDC Heart Disease Dataset (Kaggle)  
- Methodology: Meta-ensemble (stacking) machine learning techniques  
- Explainable AI: SHAP & LIME frameworks  
