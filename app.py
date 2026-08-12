import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Absenteeism Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 30px;
    border-radius: 15px;
    background-color: white;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
}

.result-number {
    font-size: 45px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">👨‍💼 Employee Absenteeism Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Random Forest Regression Model</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

try:

    df = pd.read_csv("Employee.csv")

except FileNotFoundError:

    st.error(
        "❌ Employee.csv file not found. "
        "Please keep Employee.csv in the same folder as app.py."
    )

    st.stop()


# =========================================================
# TRAIN MODEL
# =========================================================

target_column = "AbsenteeismDays"

if target_column not in df.columns:

    st.error(
        f"❌ Target column '{target_column}' not found."
    )

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.stop()


# Remove duplicates
df = df.drop_duplicates()

# Remove empty rows
df = df.dropna(how="all")


# =========================================================
# REMOVE ID COLUMN
# =========================================================

if "EmployeeID" in df.columns:

    df = df.drop(columns=["EmployeeID"])


# =========================================================
# TARGET
# =========================================================

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

df = df.dropna(
    subset=[target_column]
)


X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns


for col in numeric_columns:

    X[col] = X[col].fillna(
        X[col].median()
    )


categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns


for col in categorical_columns:

    if X[col].isnull().any():

        X[col] = X[col].fillna(
            X[col].mode()[0]
        )


# =========================================================
# ENCODING
# =========================================================

encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = encoder


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success(
    "Random Forest Regression"
)

st.sidebar.write(
    f"📊 Dataset Rows: {len(df)}"
)

st.sidebar.write(
    f"🌲 Number of Trees: 200"
)

st.sidebar.write(
    f"🎯 Target: {target_column}"
)


# =========================================================
# MODEL METRICS
# =========================================================

st.subheader("📊 Model Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "MAE",
        round(mae, 2)
    )


with col2:

    st.metric(
        "RMSE",
        round(rmse, 2)
    )


with col3:

    st.metric(
        "R² Score",
        round(r2, 4)
    )


with col4:

    st.metric(
        "Training Rows",
        len(X_train)
    )


st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("👤 Employee Details")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# AGE
# ---------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )


# ---------------------------------------------------------
# GENDER
# ---------------------------------------------------------

with col2:

    gender_options = (
        encoders["Gender"].classes_
        if "Gender" in encoders
        else ["Male", "Female"]
    )

    gender = st.selectbox(
        "Gender",
        gender_options
    )


# ---------------------------------------------------------
# DEPARTMENT
# ---------------------------------------------------------

with col3:

    department_options = (
        encoders["Department"].classes_
        if "Department" in encoders
        else ["HR", "IT", "Sales"]
    )

    department = st.selectbox(
        "Department",
        department_options
    )


col4, col5, col6 = st.columns(3)


# ---------------------------------------------------------
# JOB ROLE
# ---------------------------------------------------------

with col4:

    jobrole_options = (
        encoders["JobRole"].classes_
        if "JobRole" in encoders
        else ["Manager", "Developer", "Analyst"]
    )

    job_role = st.selectbox(
        "Job Role",
        jobrole_options
    )


# ---------------------------------------------------------
# MONTHLY INCOME
# ---------------------------------------------------------

with col5:

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=30000,
        step=1000
    )


# ---------------------------------------------------------
# YEARS AT COMPANY
# ---------------------------------------------------------

with col6:

    years_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5
    )


col7, col8, col9 = st.columns(3)


# ---------------------------------------------------------
# JOB SATISFACTION
# ---------------------------------------------------------

with col7:

    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=5,
        value=3
    )


# ---------------------------------------------------------
# WORK HOURS
# ---------------------------------------------------------

with col8:

    work_hours = st.number_input(
        "Work Hours Per Week",
        min_value=1,
        max_value=100,
        value=40
    )


# ---------------------------------------------------------
# DISTANCE
# ---------------------------------------------------------

with col9:

    distance = st.number_input(
        "Distance From Home (km)",
        min_value=0,
        max_value=200,
        value=10
    )


col10, col11 = st.columns(2)


# ---------------------------------------------------------
# OVERTIME
# ---------------------------------------------------------

with col10:

    overtime_options = (
        encoders["OverTime"].classes_
        if "OverTime" in encoders
        else ["Yes", "No"]
    )

    overtime = st.selectbox(
        "OverTime",
        overtime_options
    )


# ---------------------------------------------------------
# WORK LIFE BALANCE
# ---------------------------------------------------------

with col11:

    work_life_balance = st.slider(
        "Work Life Balance",
        min_value=1,
        max_value=5,
        value=3
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Absenteeism",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = {}

    for feature in X.columns:

        if feature == "Age":

            input_data[feature] = age

        elif feature == "Gender":

            input_data[feature] = encoders["Gender"].transform(
                [gender]
            )[0]

        elif feature == "Department":

            input_data[feature] = encoders["Department"].transform(
                [department]
            )[0]

        elif feature == "JobRole":

            input_data[feature] = encoders["JobRole"].transform(
                [job_role]
            )[0]

        elif feature == "MonthlyIncome":

            input_data[feature] = monthly_income

        elif feature == "YearsAtCompany":

            input_data[feature] = years_company

        elif feature == "JobSatisfaction":

            input_data[feature] = job_satisfaction

        elif feature == "WorkHoursPerWeek":

            input_data[feature] = work_hours

        elif feature == "DistanceFromHome":

            input_data[feature] = distance

        elif feature == "OverTime":

            input_data[feature] = encoders["OverTime"].transform(
                [overtime]
            )[0]

        elif feature == "WorkLifeBalance":

            input_data[feature] = work_life_balance

        else:

            input_data[feature] = 0


    input_df = pd.DataFrame(
        [input_data],
        columns=X.columns
    )


    prediction = model.predict(
        input_df
    )[0]


    prediction = max(
        0,
        prediction
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📈 Prediction Result"
    )

    st.markdown(
        f'<div class="result-number">{prediction:.2f}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Predicted Absenteeism Days"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    if prediction <= 2:

        st.success(
            "🟢 Low expected absenteeism"
        )

    elif prediction <= 5:

        st.warning(
            "🟡 Moderate expected absenteeism"
        )

    else:

        st.error(
            "🔴 High expected absenteeism"
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

with st.expander("📌 Feature Importance"):

    importance_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

with st.expander("📊 Actual vs Predicted"):

    result_df = pd.DataFrame({

        "Actual Absenteeism Days":
            y_test.values,

        "Predicted Absenteeism Days":
            np.round(y_pred, 2)

    })

    st.dataframe(
        result_df.head(20),
        use_container_width=True
    )