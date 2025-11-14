import joblib
import xgboost as xgb

models = {
    "Random Forest": joblib.load("models/rf.pkl"),
    "Extra Trees": joblib.load("models/ert.pkl")  # corrected, assuming ert.pkl exists
}

xgb_model = xgb.XGBClassifier()  # Or XGBRegressor depending on your model type
xgb_model.load_model("models/xgb_model.json")

# Add XGBoost model to the models dict
models["XGBoost"] = xgb_model
