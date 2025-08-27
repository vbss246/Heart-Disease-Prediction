from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

class StackingClassifierPipeline:
    def __init__(self, models):
        estimators = [(name, mod) for name, mod in models.items()]
        self.model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, passthrough=True)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model(self):
        return self.model
