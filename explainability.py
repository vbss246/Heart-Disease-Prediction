import shap

def explain_model(stacker, X_test):
    print("\nExplainability with SHAP values:")
    # SHAP only supports some models, so use base estimators for explanation.
    for name, estimator in stacker.get_model().named_estimators_.items():
        try:
            explainer = shap.Explainer(estimator, X_test)
            shap_values = explainer(X_test)
            print(f"\nExplaining: {name}")
            shap.summary_plot(shap_values, X_test, show=False)  # show=False for scripts
        except Exception as e:
            print(f"Could not explain {name}: {e}")
