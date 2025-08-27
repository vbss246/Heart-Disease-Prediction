from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def get_models():
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }
    return models
