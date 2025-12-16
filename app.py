import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Прогноз дохода (GSS)", layout="centered")
st.title("📈 Прогноз годового дохода по данным GSS")
st.markdown("На основе модели **Случайного леса**, обученной на данных General Social Survey (GSS).")

# === ЗАГРУЗКА МОДЕЛИ ===
model_path = "random_forest_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Файл модели `random_forest_model.pkl` не найден. Поместите его в эту папку.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# === ВВОД ДАННЫХ ===
st.subheader("1. Введите данные респондента")

# Основные переменные из анкеты
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Возраст (age)", min_value=16, max_value=99, value=35)
    gender = st.selectbox("Пол (gender)", ["Male", "Female"])
    educcat = st.selectbox("Уровень образования (educcat)", [
        "Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"
    ])

with col2:
    marital = st.selectbox("Семейное положение (marital)", [
        "Married", "Never Married", "Divorced", "Separated", "Widowed"
    ])
    wrkstat = st.selectbox("Трудоустройство (wrkstat)", [
        "Full-Time", "Part-Time", "Temporarily Not Working", "Unemployed, Laid Off",
        "Retired", "Housekeeper", "School", "Other"
    ])
    prestg10 = st.slider("Престиж профессии (prestg10)", 0, 100, 45)

childs = st.number_input("Количество детей (childs)", min_value=0, max_value=20, value=2)

# === ПРЕОБРАЗОВАНИЕ В ПРИЗНАКИ ===

# is_male
is_male = 1 if gender == "Male" else 0

# education_num (примерное соответствие)
edu_map = {
    "Less Than High School": 10,
    "High School": 12,
    "Junior College": 14,
    "Bachelor": 16,
    "Graduate": 18
}
education_num = edu_map[educcat]

# is_employed
employed_statuses = ["Full-Time", "Part-Time", "Temporarily Not Working"]
is_employed = 1 if wrkstat in employed_statuses else 0

# work_experience: приблизительно как (возраст - образование - 6)
work_experience = age - (education_num + 6)
work_experience = max(work_experience, 0)

# Производные
age_squared = age ** 2
experience_squared = work_experience ** 2

# One-hot для marital (Divorced — базовый)
marital_Married = 1 if marital == "Married" else 0
marital_Never_Married = 1 if marital == "Never Married" else 0
marital_Widowed = 1 if marital == "Widowed" else 0
marital_Separated = 1 if marital == "Separated" else 0

# === СОБРАНИЕ ВЕКТОРА ===
input_df = pd.DataFrame([{
    "prestg10": prestg10,
    "education_num": education_num,
    "is_male": is_male,
    "childs": childs,
    "age": age,
    "experience_squared": experience_squared,
    "work_experience": work_experience,
    "age_squared": age_squared,
    "marital_Married": marital_Married,
    "is_employed": is_employed,
    "marital_Never Married": marital_Never_Married,
    "marital_Widowed": marital_Widowed,
    "marital_Separated": marital_Separated
}])

# === ПРЕДСКАЗАНИЕ ===
if st.button("🔍 Предсказать доход"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"**Прогнозируемый годовой доход: ${pred:,.2f}**")
        st.info("Модель лучше всего предсказывает средние значения дохода (см. отчёт).")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        st.write("Убедитесь, что модель обучена на тех же 13 признаках.")