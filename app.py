import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Streamlit ML App", layout="wide")
st.title("🤖 Machine Learning Web Application")
st.markdown("This application allows you to upload or select a dataset, preprocess it, train a model, and make predictions.")

# Data Source Selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose your data source:", ["Upload your data", "Use example dataset"])

if data_source == "Upload your data":
    file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "tsv"])
    if file:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        elif file.name.endswith("xlsx"):
            df = pd.read_excel(file)
        elif file.name.endswith("tsv"):
            df = pd.read_csv(file, sep='\t')
        else:
            st.error("Unsupported file format")
            st.stop()
    else:
        st.warning("Please upload a file to proceed.")
        st.stop()
else:
    dataset_name = st.sidebar.selectbox("Select a sample dataset", ["titanic", "tips", "iris"])
    df = sns.load_dataset(dataset_name)

# Display Basic Info
st.subheader("🔍 Dataset Preview")
st.write(df.head())
st.write("Shape:", df.shape)
st.write("Description:", df.describe())
st.write("Column Names:", df.columns.tolist())

# Feature and Target Selection
features = st.multiselect("Select feature columns", df.columns.tolist())
target = st.selectbox("Select target column", df.columns.tolist())

if not features or not target:
    st.warning("Please select feature and target columns.")
    st.stop()

# Problem Type Detection
if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
    problem_type = "regression"
    st.info("Detected Regression Problem")
else:
    problem_type = "classification"
    st.info("Detected Classification Problem")

# Preprocessing
data = df[features + [target]].copy()
imputer = IterativeImputer()
scaler = StandardScaler()
label_encoders = {}

num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Impute numeric
data[num_cols] = pd.DataFrame(imputer.fit_transform(data[num_cols]), columns=num_cols)

# Encode categorical
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

X = data[features]
y = data[target]
if isinstance(y, pd.DataFrame):
    y = y.squeeze()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-Test Split
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=50, value=20, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Model Selection
st.sidebar.header("Model Selection")
if problem_type == "classification":
    model_name = st.sidebar.selectbox("Choose Model", ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SupportVectorMachine", "KNeighbors"])
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "SupportVectorMachine": SVC(probability=True),
        "KNeighbors": KNeighborsClassifier()
    }
else:
    model_name = st.sidebar.selectbox("Choose Model", ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR", "KNeighbors"])
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "KNeighbors": KNeighborsRegressor()
    }

model = models[model_name]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📊 Model Evaluation")
if problem_type == "classification":
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"**Accuracy**: {acc:.4f}")
    st.write(f"**Precision**: {prec:.4f}")
    st.write(f"**Recall**: {rec:.4f}")
    st.write(f"**F1 Score**: {f1:.4f}")
    st.write("**Confusion Matrix:**")
    st.write(cm)

    # ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
        st.write("**ROC Curve:**")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
else:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**MSE**: {mse:.4f}")
    st.write(f"**MAE**: {mae:.4f}")
    st.write(f"**RMSE**: {rmse:.4f}")
    st.write(f"**R2 Score**: {r2:.4f}")

# Save Model
if st.checkbox("💾 Save Model"):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model saved as best_model.pkl")

# Prediction
if st.checkbox("🔮 Make Prediction"):
    st.write("Provide input data for prediction")
    input_data = []
    for col in features:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    input_df = pd.DataFrame([input_data], columns=features)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    if target in label_encoders:
        pred = label_encoders[target].inverse_transform(pred.astype(int))
    st.write(f"### 🎯 Prediction Result: {pred[0]}")