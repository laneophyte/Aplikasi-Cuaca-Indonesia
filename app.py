import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Fungsi untuk mendapatkan data cuaca dari OpenWeatherMap
def get_weather(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},ID&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

# Konfigurasi API
api_key = "0b177fb6f5bd23c0f6d36429a8c2ba99"  

# Simulasi data historis untuk melatih model prediksi
data = pd.DataFrame({
    'kelembapan': [60, 65, 70, 75, 80, 85, 90],
    'tekanan': [1000, 1005, 1010, 1015, 1020, 1025, 1030],
    'suhu_max': [30, 31, 32, 33, 34, 35, 36]
})

# Membuat model Linear Regression
X = data[['kelembapan', 'tekanan']]
y = data['suhu_max']
model = LinearRegression().fit(X, y)

# Konfigurasi font untuk grafik
rcParams['font.family'] = 'DejaVu Sans'  # Ganti dengan nama font yang diinginkan
rcParams['font.size'] = 12

# Judul aplikasi
st.title("Aplikasi Cuaca Indonesia dengan Prediksi")
st.write("Dapatkan informasi cuaca terkini dan prediksi suhu maksimum di berbagai kota di Indonesia.")

# Input lokasi dari pengguna
city = st.text_input("Masukkan nama kota di Indonesia:", "Jakarta")

# Mendapatkan data cuaca
if city:
    st.write(f"Menampilkan cuaca untuk: **{city}**")
    weather_data = get_weather(city, api_key)
    
    if weather_data:
        # Menampilkan data cuaca dari API
        kelembapan = weather_data['main']['humidity']
        tekanan = weather_data['main']['pressure']
        suhu_sekarang = weather_data['main']['temp']
        suhu_min = weather_data['main']['temp_min']
        suhu_max = weather_data['main']['temp_max']
        kondisi = weather_data['weather'][0]['description'].capitalize()

        st.write(f"🌡️ Suhu Sekarang: {suhu_sekarang}°C")
        st.write(f"💧 Kelembapan: {kelembapan}%")
        st.write(f"🌬️ Kecepatan Angin: {weather_data['wind']['speed']} m/s")
        st.write(f"🌤️ Kondisi: {kondisi}")
        
        # Prediksi suhu maksimum menggunakan model
        suhu_prediksi = model.predict([[kelembapan, tekanan]])[0]
        st.write(f"📈 Prediksi Suhu Maksimum (berdasarkan kelembapan & tekanan): {suhu_prediksi:.2f}°C")
        
        # Visualisasi data dengan font yang dimodifikasi
        temps = {
            "Suhu Maksimum API": suhu_max,
            "Suhu Minimum": suhu_min,
            "Suhu Sekarang": suhu_sekarang,
            "Prediksi Suhu Maksimum": suhu_prediksi
        }
        plt.figure(figsize=(6, 4))
        plt.bar(temps.keys(), temps.values(), color=['red', 'blue', 'orange', 'green'])
        plt.title(f"Grafik Suhu di {city}", fontsize=15, fontweight='bold')
        plt.ylabel("Suhu (°C)", fontsize=12)
        plt.xlabel("Kategori", fontsize=12)
        plt.xticks(rotation=0, fontsize=7)
        plt.yticks(fontsize=7)
        st.pyplot(plt)
    else:
        st.error("Gagal mendapatkan data cuaca. Pastikan nama kota benar dan coba lagi.")
else:
    st.info("Masukkan nama kota untuk melihat cuaca.")
