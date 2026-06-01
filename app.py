import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Stacking Regressor", layout="wide")

st.title("Stacking Regressor with Diabetes Dataset")

st.header("1. Data Ingestion")

@st.cache_data
def load_data():
    data = load_diabetes(as_frame=True)
    df = data.frame
    raw_path = os.path.join(RAW_DIR, "diabetes_dataset.csv")
    df.to_csv(raw_path, index=False)
    np.random.seed(42)
    for col in df.columns[:-1]: df.loc[df.sample(frac=0.1).index, col] = np.nan
    return df

df = load_data()

st.success("Diabetes Dataset Loaded Successfully")

st.dataframe(df, use_container_width=True)

st.header("2. Data Cleaning")

strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Most Frequent", "Drop Rows"])

df_clean = df.copy()

if strategy == "Drop Rows":
    df_clean = df_clean.dropna()
else:
    fill_map = {"Mean": "mean", "Median": "median", "Most Frequent": "most_frequent"}
    imputer = SimpleImputer(strategy=fill_map[strategy])
    cols = df_clean.select_dtypes(include=np.number).columns
    df_clean[cols] = imputer.fit_transform(df_clean[cols])

st.dataframe(df_clean, use_container_width=True)

if st.button("Save Cleaned Dataset"):
    path = os.path.join(CLEAN_DIR, "cleaned_diabetes_dataset.csv")
    df_clean.to_csv(path, index=False)
    st.success("Dataset Saved Successfully")

st.header("3. Load Cleaned Dataset")

files = [f for f in os.listdir(CLEAN_DIR) if "diabetes_dataset" in f]

if not files:
    st.warning("No cleaned dataset found")
    st.stop()

file = st.selectbox("Select Dataset", files)

data = pd.read_csv(os.path.join(CLEAN_DIR, file))

st.dataframe(data, use_container_width=True)

st.sidebar.header("Model Settings")

number_of_random_search_iterations = st.sidebar.slider("Number Of Randomized Search Iterations", 5, 50, 10)
number_of_cross_validation_folds = st.sidebar.slider("Number Of Cross Validation Folds", 2, 10, 5)
test_dataset_size = st.sidebar.slider("Test Dataset Size", 0.1, 0.5, 0.25)
random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)
evaluation_metric = st.sidebar.selectbox("Evaluation Metric", ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"])

X = data.drop(columns=["target"])
y = data["target"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_dataset_size, random_state=random_seed)

base_models = [("rf", RandomForestRegressor(random_state=random_seed)), ("gb", GradientBoostingRegressor(random_state=random_seed))]

meta_model = LinearRegression()

stack_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=number_of_cross_validation_folds, n_jobs=-1)

parameter_grid = {"rf__n_estimators": [50, 100, 200], "rf__max_depth": [3, 5, 10, None], "gb__n_estimators": [50, 100, 200], "gb__learning_rate": [0.01, 0.05, 0.1]}

st.header("4. Randomized Search CV Training")

random_search = RandomizedSearchCV(estimator=stack_model, param_distributions=parameter_grid, n_iter=number_of_random_search_iterations, cv=number_of_cross_validation_folds, verbose=1, random_state=random_seed, n_jobs=-1, scoring=evaluation_metric)

random_search.fit(X_train, y_train)

model = random_search.best_estimator_

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.success(f"R² Score: {r2:.4f}")

st.write("Mean Absolute Error:", mae)

st.write("Root Mean Squared Error:", rmse)

st.write("Best Hyperparameters:", random_search.best_params_)

st.write("Best Cross Validation Score:", random_search.best_score_)

st.header("Model Information")

st.subheader("Selected Model")

st.write(type(model).__name__)

st.subheader("Base Learners")

base_learner_df = pd.DataFrame({
    "Learner Name": list(model.named_estimators_.keys()),
    "Model Type": [type(est).__name__ for est in model.named_estimators_.values()]
})

st.dataframe(base_learner_df, use_container_width=True)

st.subheader("Meta Learner")

meta_learner_df = pd.DataFrame({
    "Meta Learner": [type(model.final_estimator_).__name__]
})

st.dataframe(meta_learner_df, use_container_width=True)

st.subheader("Best Parameters Found")

best_params_df = pd.DataFrame(list(random_search.best_params_.items()), columns=["Parameter", "Value"])

st.dataframe(best_params_df, use_container_width=True)

st.header("5. Regression Metrics")

metrics_df = pd.DataFrame({
    "Metric": ["R² Score", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

st.dataframe(metrics_df, use_container_width=True)

st.header("6. Actual vs Predicted")

comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})

fig, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(data=comparison_df, x="Actual", y="Predicted", ax=ax)

ax.plot([comparison_df["Actual"].min(), comparison_df["Actual"].max()], [comparison_df["Actual"].min(), comparison_df["Actual"].max()], color="red")

st.pyplot(fig)

st.header("7. Feature Importance")

rf_model = model.named_estimators_["rf"]

feature_importance_dataframe = pd.DataFrame({"Feature": data.drop(columns=["target"]).columns, "Importance": rf_model.feature_importances_})

feature_importance_dataframe = feature_importance_dataframe.sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(data=feature_importance_dataframe, x="Importance", y="Feature", palette="viridis", ax=ax2)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

st.dataframe(feature_importance_dataframe, use_container_width=True)

st.header("8. Save Model")

model_path = os.path.join(MODEL_DIR, "stacking_regressor.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

st.success(f"Model Saved At: {model_path}")

st.header("9. Sample Predictions")

sample_predictions = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": predictions[:10]})

st.dataframe(sample_predictions, use_container_width=True)