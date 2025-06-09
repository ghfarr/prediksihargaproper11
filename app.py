
import streamlit as st
import pickle
import numpy as np

# Load model dan label encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")
st.title("🏡 Prediksi Harga Properti")
st.write("Aplikasi cerdas untuk memprediksi harga properti berdasarkan fitur yang Anda input.")

# Input pengguna
luas_bangunan = st.number_input("Luas Bangunan (GrLivArea)", min_value=10, step=1)
luas_tanah = st.number_input("Luas Tanah (LotArea)", min_value=10, step=1)
kamar_tidur = st.number_input("Jumlah Kamar Tidur (BedroomAbvGr)", min_value=1, step=1)
tahun_dibangun = st.number_input("Tahun Dibangun (YearBuilt)", min_value=1800, max_value=2100, step=1)
kualitas = st.slider("Kualitas Bangunan (OverallQual)", 1, 10, 5)
garasi = st.slider("Jumlah Mobil di Garasi (GarageCars)", 0, 5, 2)
kamar_mandi = st.slider("Jumlah Kamar Mandi (FullBath)", 0, 4, 1)

# Lokasi properti dengan nama-nama provinsi
provinsi_labels = label_encoders['Provinsi'].classes_.tolist()
provinsi = st.selectbox("Lokasi Properti (Provinsi)", provinsi_labels)

if st.button("Prediksi Harga"):
    try:
        provinsi_encoded = label_encoders['Provinsi'].transform([provinsi])[0]
        input_data = np.array([
            [luas_bangunan, luas_tanah, kamar_tidur, tahun_dibangun,
             kualitas, garasi, kamar_mandi, provinsi_encoded]
        ])
        harga_prediksi = model.predict(input_data)[0]
        harga_prediksi *= 1000  # jika model hasilnya masih dalam ribuan
        harga_format = f"Rp {harga_prediksi:,.0f}".replace(",", ".")
        st.success(f"💰 Estimasi Harga Properti: {harga_format}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
