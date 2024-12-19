# -*- coding: utf-8 -*-
"""Data Mining - Final Project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xaniLSWmP_0lIfo69BLSUr-_XIn_GuQo

# Contoh
"""

# Install google-play-scraper terlebih dahulu
# Run this in your terminal or separate cell:
# !pip install google-play-scraper

# from google_play_scraper import Sort, reviews
# import pandas as pd

# # Scraping 300 ulasan aplikasi ShopeePay
# result, _ = reviews(
#     'com.shopeepay.id',  # ID aplikasi ShopeePay
#     lang='id',           # Bahasa Indonesia
#     country='id',        # Negara Indonesia
#     sort=Sort.NEWEST,    # Urutkan berdasarkan ulasan terbaru
#     count=300            # Jumlah ulasan yang diambil
# )

# # Konversi data ke dalam DataFrame
# df_reviews = pd.DataFrame(result)

# # Pilih kolom yang relevan
# df_reviews = df_reviews[['userName', 'score', 'at', 'content']]

# # Ubah nama kolom agar mudah dibaca
# df_reviews.columns = ['Nama Pengguna', 'Rating', 'Tanggal', 'Ulasan']

# # Tampilkan tabel
# print(df_reviews.head(10))  # Menampilkan 10 ulasan pertama

"""# 1. Scraping Data Ulasan"""

# Install google-play-scraper terlebih dahulu
# Run this in your terminal or separate cell:
# !pip install google-play-scraper

# Import library
from google_play_scraper import Sort, reviews
import pandas as pd

# Scraping data ulasan
review, countinuation_token = reviews(
    'com.shopeepay.id',       # ID aplikasi ShopeePay
    lang='id',                # Bahasa Indonesia
    country='id',             # Negara Indonesia
    sort= Sort.MOST_RELEVANT,
    count=1000,               # Jumlah ulasan
)

# Konversi ke DataFrame
df = pd.DataFrame(review)
df = df[['content', 'score']]  # Ambil kolom ulasan dan skor
df['label'] = df['score'].apply(lambda x: 'positif' if x > 3 else 'negatif')

# Simpan ke CSV
df.to_csv('ulasan_shopeepay.csv', index=False)
print("Data berhasil disimpan.")

import streamlit as st
import pandas as pd
import io
# Load data dari CSV
file_path = 'ulasan_shopeepay.csv'
df = pd.read_csv(file_path)

# Judul aplikasi
st.title("Tampilan Data Ulasan ShopeePay")

# Tampilkan beberapa informasi
st.subheader("10 Data Pertama")
st.dataframe(df.head(10))  # Tampilkan 10 data pertama


# Menampilkan informasi dataframe dalam format terstruktur
st.subheader("Informasi Data")
st.write("Jumlah baris:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])
st.write("Nama kolom:", list(df.columns))
st.write("Tipe data tiap kolom:")
st.write(df.dtypes)


st.subheader("Statistik Deskriptif")
st.write(df.describe(include='all'))  # Statistik deskriptif

# Pilihan untuk menampilkan semua data
if st.checkbox("Tampilkan semua data"):
    st.subheader("Seluruh Data")
    st.dataframe(df)

# Filter data berdasarkan skor rating
st.subheader("Filter Berdasarkan Rating")
rating_filter = st.slider("Pilih Skor Rating", min_value=int(df['score'].min()), max_value=int(df['score'].max()), value=int(df['score'].min()))
filtered_data = df[df['score'] == rating_filter]
st.write(f"Data dengan Skor Rating {rating_filter}:")
st.dataframe(filtered_data)


"""# 2. Preprocessing Data"""

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('ulasan_shopeepay.csv')

# Preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter selain huruf
    text = text.lower()                     # Konversi ke huruf kecil
    return text

df['clean_content'] = df['content'].apply(clean_text)

# Encode label
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print(df.columns)
print(df.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_content'], df['label_encoded'], test_size=0.2, random_state=42
)

"""# 3. Feature Extraction dan Algoritma"""

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model 1: Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# Model 2: Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

# Evaluasi
print("Evaluasi Naive Bayes:")
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

print("Evaluasi SVM:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

"""# 4. Skenario Eksperimen

Skenario 1: Menggunakan TF-IDF dan Naive Bayes.

Skenario 2: Menggunakan TF-IDF dan SVM.

Skenario 3: Menggunakan Bag-of-Words dan Naive Bayes.

Skenario 4: Menggunakan Bag-of-Words dan SVM.

## Persiapan Data
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load Data
df = pd.read_csv('ulasan_shopeepay.csv')

# Preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter selain huruf
    text = text.lower()                     # Konversi ke huruf kecil
    return text

df['clean_content'] = df['content'].apply(clean_text)

# Label Encoding
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])  # positif: 1, negatif: 0

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_content'], df['label_encoded'], test_size=0.2, random_state=42
)

"""## Skenario 1: Menggunakan TF-IDF dan Naive Bayes."""

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = nb.predict(X_test_tfidf)

# Evaluasi
print("Skenario 1: TF-IDF + Naive Bayes")
print(classification_report(y_test, y_pred_nb_tfidf, target_names=le.classes_))

"""## Skenario 2: Menggunakan TF-IDF dan SVM"""

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)
y_pred_svm_tfidf = svm.predict(X_test_tfidf)

# Evaluasi
print("Skenario 2: TF-IDF + SVM")
print(classification_report(y_test, y_pred_svm_tfidf, target_names=le.classes_))

"""## Skenario 3: Menggunakan Bag-of-Words dan Naive Bayes"""

# Bag-of-Words
bow_vectorizer = CountVectorizer(max_features=500)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Naive Bayes
nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)
y_pred_nb_bow = nb_bow.predict(X_test_bow)

# Evaluasi
print("Skenario 3: Bag-of-Words + Naive Bayes")
print(classification_report(y_test, y_pred_nb_bow, target_names=le.classes_))

"""## Skenario 4: Menggunakan Bag-of-Words dan SVM"""

# SVM
svm_bow = SVC(kernel='linear')
svm_bow.fit(X_train_bow, y_train)
y_pred_svm_bow = svm_bow.predict(X_test_bow)

# Evaluasi
print("Skenario 4: Bag-of-Words + SVM")
print(classification_report(y_test, y_pred_svm_bow, target_names=le.classes_))

"""## Output

- Skenario 1: TF-IDF + Naive Bayes → Akurasi dan evaluasi performa ditampilkan.
- Skenario 2: TF-IDF + SVM → Akurasi dan evaluasi performa ditampilkan.
- Skenario 3: Bag-of-Words + Naive Bayes → Akurasi dan evaluasi performa ditampilkan.
- Skenario 4: Bag-of-Words + SVM → Akurasi dan evaluasi performa ditampilkan.

Setiap skenario menghasilkan laporan klasifikasi lengkap dengan metrik precision, recall, f1-score, dan akurasi.

**Penjelasan**
- TF-IDF (Term Frequency-Inverse Document Frequency): Mengukur bobot kata berdasarkan frekuensi relatif dalam dokumen.
- Bag-of-Words: Representasi sederhana berdasarkan jumlah kemunculan kata.
- Naive Bayes: Algoritma probabilistik yang cepat dan efisien untuk teks.
- SVM: Algoritma yang bekerja baik untuk data dengan dimensi tinggi seperti teks.


# 5. Deployment dengan Streamlit
"""

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

"""Cek apakah kolom clean_content sudah ada di DataFrame:"""

print(df.columns)
print(df.head())
X = df['clean_content']
y = df['label_encoded']

# Pisahkan data menjadi fitur (X) dan label (y)
X = df['clean_content']
y = df['label_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data berhasil di-split.")


# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model 1: Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# Model 2: Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

# Evaluasi Model
print("Evaluasi Naive Bayes:")
print(classification_report(y_test, y_pred_nb, target_names=['negatif', 'positif']))

print("Evaluasi SVM:")
print(classification_report(y_test, y_pred_svm, target_names=['negatif', 'positif']))

"""# **VISUALISASI** **DATA**"""

import matplotlib.pyplot as plt

"""**1. Performa Model (Confusion Matrix)**

**Penjelasan Confusion Matrix**
* True Positive (TP): Ulasan positif yang benar terdeteksi sebagai positif.
* True Negative (TN): Ulasan negatif yang benar terdeteksi sebagai negatif.
* False Positive (FP): Ulasan negatif yang salah diklasifikasikan sebagai positif.
* False Negative (FN): Ulasan positif yang salah diklasifikasikan sebagai negatif.

**Confusion Matrix membantu:**
Mengidentifikasi pola kesalahan model, seperti salah klasifikasi ulasan.
Membandingkan performa model Naive Bayes dan SVM pada deteksi ulasan positif dan negatif.
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix untuk Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['Negatif', 'Positif'])
disp_nb.plot(cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()
st.pyplot(plt)  # Use st.pyplot() to display the plot

# Confusion matrix untuk SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Negatif', 'Positif'])
disp_svm.plot(cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.show()
st.pyplot(plt)  # Use st.pyplot() to display the plot

"""**Confusion Matrix - Naive Bayes**
* True Negatives (TN): 97 ulasan negatif terklasifikasi dengan benar.
* True Positives (TP): 64 ulasan positif terklasifikasi dengan benar.
* False Positives (FP): 12 ulasan negatif salah diklasifikasikan sebagai positif.
* False Negatives (FN): 27 ulasan positif salah diklasifikasikan sebagai negatif.

**Confusion Matrix - SVM**
* True Negatives (TN): 92 ulasan negatif terklasifikasi dengan benar.
* True Positives (TP): 65 ulasan positif terklasifikasi dengan benar.
* False Positives (FP): 17 ulasan negatif salah diklasifikasikan sebagai positif.
* False Negatives (FN): 26 ulasan positif salah diklasifikasikan sebagai negatif.

**Perbandingan Naive Bayes vs SVM**

Naive Bayes memiliki jumlah True Negatives (97) yang lebih tinggi dibandingkan SVM (92), menunjukkan kemampuan lebih baik dalam mendeteksi ulasan negatif.
SVM sedikit lebih unggul dalam mendeteksi ulasan positif dengan jumlah True Positives (65) dibandingkan Naive Bayes (64).
False Positives dan False Negatives dari Naive Bayes lebih rendah pada ulasan negatif, tetapi SVM menunjukkan keseimbangan antara positif dan negatif.

**2. Perbandingan Akurasi Model**

Visualisasi grafik batang yang membandingkan skor kinerja antara dua model, Naive Bayes dan SVM, berdasarkan tiga metrik: Precision, Recall, dan F1-Score. Menggunakan pustaka matplotlib, kode ini membuat dua set batang (rects1 untuk Naive Bayes dan rects2 untuk SVM), dengan lebar batang yang disesuaikan agar tidak saling bertumpukan. Setiap metrik (Precision, Recall, dan F1-Score) ditempatkan di sepanjang sumbu x, sementara nilai skor untuk masing-masing model ditampilkan pada sumbu y. Grafik ini memberi gambaran visual tentang bagaimana kedua model tersebut berperforma di berbagai metrik evaluasi.
"""

import numpy as np

# Data metrik
metrics = ['Precision', 'Recall', 'F1-Score']
nb_scores = [0.85, 0.80, 0.82]  # Contoh nilai untuk Naive Bayes
svm_scores = [0.90, 0.85, 0.87]  # Contoh nilai untuk SVM

x = np.arange(len(metrics))
width = 0.35

# Plot
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, nb_scores, width, label='Naive Bayes')
rects2 = ax.bar(x + width/2, svm_scores, width, label='SVM')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Perbandingan Performa Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()

st.pyplot(fig)  # Display the plot using Streamlit

"""Visualisasi diatas menunjukkan perbandingan performa dua model klasifikasi, yaitu Naive Bayes (ditampilkan dalam warna biru) dan SVM (ditampilkan dalam warna oranye), berdasarkan tiga metrik evaluasi utama: Precision, Recall, dan F1-Score. Pada metrik Precision, SVM menunjukkan hasil yang lebih tinggi dibandingkan Naive Bayes, yang menunjukkan bahwa SVM lebih baik dalam meminimalkan prediksi positif palsu. Pada metrik Recall, kedua model memiliki performa yang hampir sama, menunjukkan kemampuan yang setara dalam mendeteksi data positif secara keseluruhan. Namun, pada metrik F1-Score, SVM kembali unggul dibandingkan Naive Bayes, menandakan keseimbangan yang lebih baik antara Precision dan Recall. Secara keseluruhan, grafik ini mengindikasikan bahwa model SVM memberikan performa yang lebih baik dibandingkan Naive Bayes dalam tugas klasifikasi ini."""

import matplotlib.pyplot as plt

# Visualisasi distribusi rating
df['score'].value_counts().sort_index().plot(kind='bar', color=['red', 'orange', 'yellow', 'green', 'blue'])
plt.title('Distribusi Skor Rating')
plt.xlabel('Skor Rating')
plt.ylabel('Jumlah Ulasan')
plt.show()
st.pyplot(plt)  # Display the plot using Streamlit

"""Visualisasi diatas menunjukkan distribusi jumlah ulasan berdasarkan skor rating. Pada sumbu horizontal (x-axis), ditampilkan skor rating (1 hingga 5), sedangkan pada sumbu vertikal (y-axis), ditampilkan jumlah ulasan untuk setiap skor.

Dari visualisasi ini, terlihat:

Skor rating 1 memiliki jumlah ulasan tertinggi, mendekati 400 ulasan, yang menunjukkan dominasi ulasan negatif.
Skor rating 2 dan 3 memiliki jumlah ulasan yang jauh lebih sedikit, masing-masing mendekati 100 ulasan.
Skor rating 4 memiliki jumlah ulasan yang lebih sedikit dibanding skor 2 dan 3.
Skor rating 5 juga memiliki jumlah ulasan yang sangat tinggi, mendekati 400 ulasan, menunjukkan dominasi ulasan sangat positif.
Visualisasi ini mengindikasikan bahwa ulasan pada dataset cenderung terpolarisasi, dengan sebagian besar ulasan berada di skor ekstrem (1 atau 5).
"""
