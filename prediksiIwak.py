import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    """Memuat model Random Forest yang telah disimpan"""
    try:
        with open('iwakRf.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model tidak ditemukan! Pastikan file 'iwakRf.pkl' ada di direktori.")
        return None

def load_label_encoder():
    """Memuat LabelEncoder yang telah disimpan"""
    try:
        with open('label_encoder.pkl', 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)
        return label_encoder
    except FileNotFoundError:
        st.error("LabelEncoder tidak ditemukan! Pastikan file 'label_encoder.pkl' ada di direktori.")
        return None

def user_input_features():
    """Fungsi untuk mendapatkan input pengguna"""
    length = st.number_input('Panjang (cm)', min_value=0.0, value=50.0, step=1.0)
    weight = st.number_input('Berat (kg)', min_value=0.0, value=10.0, step=0.1)
    w_l_ratio = st.number_input('Rasio Berat-Tinggi', min_value=0.0, value=0.2, step=0.1)

    # Menyusun data menjadi dataframe dengan kolom sesuai yang digunakan saat pelatihan
    data = {
        'length': length,  # Ganti 'Length' menjadi 'length' (huruf kecil)
        'weight': weight,  # Ganti 'Weight' menjadi 'weight' (huruf kecil)
        'w_l_ratio': w_l_ratio  # Ganti 'W_L_Ratio' menjadi 'w_l_ratio' (huruf kecil)
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Judul aplikasi
st.title('Prediksi Spesies Ikan dengan Random Forest')

# Deskripsi model
st.write("""
Model ini menggunakan algoritma **Random Forest** untuk memprediksi spesies ikan berdasarkan fitur:
- Panjang ikan (cm)
- Berat ikan (kg)
- Rasio berat-tinggi ikan

Setelah memasukkan nilai-nilai tersebut, model akan memberikan prediksi spesies ikan.
""")

# Input dari pengguna
input_df = user_input_features()

# Memuat model Random Forest
model = load_model()

# Memuat LabelEncoder
label_encoder = load_label_encoder()

# Jika model berhasil dimuat, lakukan prediksi
if model and label_encoder:
    # Melakukan prediksi kelas
    prediction = model.predict(input_df)

    # Menampilkan hasil prediksi
    st.subheader('Hasil Prediksi Spesies Ikan')
    predicted_species = prediction[0]  # Kelas spesies ikan yang diprediksi
    predicted_species_label = label_encoder.inverse_transform([predicted_species])[0]  # Mengonversi label numerik ke string

    st.write(f"Spesies yang diprediksi: {predicted_species_label}")

    # Menambahkan visualisasi probabilitas
    st.subheader('Visualisasi Probabilitas')

    # Melakukan prediksi probabilitas
    probabilities = model.predict_proba(input_df)

    # Mengubah probabilitas ke persen dan membatasi 2 desimal
    prob_percentages = probabilities * 100  # Mengubah probabilitas ke persen
    prob_percentages = prob_percentages.round(2)  # Membatasi 2 desimal

    # Membuat plot batang untuk probabilitas
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model.classes_, y=prob_percentages[0], palette="viridis")

    # Mengatur tampilan agar nama spesies ikan pada sumbu x tidak tumpang tindih
    plt.title('Probabilitas untuk Setiap Spesies Ikan')
    plt.xlabel('Spesies Ikan')
    plt.ylabel('Probabilitas (%)')

    # Mengatur posisi dan rotasi label sumbu X agar tidak numpuk
    plt.xticks(rotation=45, ha='right')  # rotasi label dan penyesuaian posisi

    # Menampilkan grafik
    st.pyplot(plt)
