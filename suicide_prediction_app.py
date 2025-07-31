
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


st.title("Prediksi Angka Kematian Bunuh Diri di AS Berdasarkan Demografi")
# baca preprocessed data
df = pd.read_csv("preprocessed_suicide_rates.csv")

df['year_diff'] = df['year'] - df['year'].min()
df_encoded = pd.get_dummies(df, columns=["age", "group"])
X = df_encoded.drop("rate", axis=1)
y = df_encoded["rate"]


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

st.sidebar.header("Input Prediksi")
future_year = st.sidebar.slider("Tahun Prediksi", min_value=2025, max_value=2050, value=2030, step=1)
age_options = df["age"].unique().tolist()
group_options = df["group"].unique().tolist()

selected_age = st.sidebar.selectbox("Kelompok Usia", age_options)
selected_group = st.sidebar.selectbox("Kelompok Demografi", group_options)


input_data = {
    "year": future_year,
    "year_diff": future_year - df["year"].min()
}

for age in df["age"].unique():
    input_data[f"age_{age}"] = 1 if age == selected_age else 0
for group in df["group"].unique():
    input_data[f"group_{group}"] = 1 if group == selected_group else 0

input_df = pd.DataFrame([input_data])


missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[X.columns]  


prediction = model.predict(input_df)[0]
st.subheader("Hasil Prediksi")
st.write(f"Perkiraan angka kematian per 100.000 penduduk pada tahun {future_year} untuk {selected_group} usia {selected_age} adalah **{prediction:.2f}**.")
