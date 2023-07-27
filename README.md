# Laporan Proyek Machine Learning - ADAM PAMUNGKAS

## Project Overview

Perkembangan teknologi yang sangat pesat dibidang digital membuat manusia dapat dengam mudah mengakses segala sesuatu yang berada di internet salah satunya multimedia. perkembangan multimedia di dunia digital membuat manusia dapat mengakses beragam vidio, musik dan buku.

perkembangan dunia digital berpengaruh terhadap perkembangan alat manusia salah satunya adalah buku, buku pada era digital ini sangat mudah untuk dicari. sistem rekomendasi membantu untuk menawarkan buku yang mungkin akan cocok dengan kriterianya. dengan adanya sistem rekomendasi buku, pengguna dapat dengan mudah memilih buku baru atau buku yang relefan dengan pengguna tanpa menghabiskan waktu.
  
## Business Understanding
### Problem Statements
- Bagaimana cara membangun model untuk memberikan rekomendasi buku yang paling cocok ?
  
### Goals
- membuat model yang dapat memberikan rekomendasi buku yang relefan kepada pengguna

### Solusi Approach
Solusi algoritma machine learning untuk sistem rekomendasi yaitu:

- Collaborative Filtering
Collaborative Filtering akan memberikan rekomendasi bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain.

## Data Understanding
Dataset yang digunakan pada proyek machine learning ini berasal dari [Book-Crossing: User review ratings](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) - kaggle.

### BX-Users.csv
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   User-ID   278858 non-null  int64  
     1   Location  278858 non-null  object 
     2   Age       168096 non-null  float64

### BX_Books.csv
     #   Column       Non-Null Count   Dtype 
    ---  ------       --------------   ----- 
     0   ISBN         271379 non-null  object
     1   Book-Title   271379 non-null  object
     2   Book-Author  271378 non-null  object
     3   Publisher    271377 non-null  object

### BX-Book-Ratings.csv
     #   Column       Non-Null Count    Dtype 
    ---  ------       --------------    ----- 
     0   User-ID      1149780 non-null  int64 
     1   ISBN         1149780 non-null  object
     2   Book-Rating  1149780 non-null  int64 

### Variable pada dataset diatas adalah sebagai berikut:
  
User-ID: id pengguna
Location: lokasi tempat tinggal
Age     : umur pengguna
ISBN    : Nomer Buku
Book-Title: judul buku
Book-Auther: kata kunci
Publisher: tempat publish
Book-Rating: nilai rating dari pengguna
  

## Data Preparation
### Menggabungkan dataset untuk mendapatkan informasi lebih dalam
Penggabungan dataset bertujuan untuk mendapatkan informasi mengenai rating buku yang bagus dan banyak disukai kebanyakan orang. 

Penghapusan variable yang tidak dibutuhkan seperti:
'Year-Of-Publication', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'

Table 1. Merge Data
|index|ISBN	      | Book-Title          |	Book-Author	         |Publisher	              |User-ID|	Book-Rating|
|-----|-----------|---------------------|----------------------|------------------------|-------|------------|
| 0	  |0195153448 |	Classical Mythology |	Mark P. O. Morford	 | Oxford University Press|2      |	0          |
| 1	  |0002005018 |	Clara Callan        |	Richard Bruce Wright | HarperFlamingo Canada	|8      |	5          |
| 2	  |0002005018 |	Clara Callan        |	Richard Bruce Wright | HarperFlamingo Canada	|11400  |	0          |
| 3	  |0002005018 |	Clara Callan        |	Richard Bruce Wright | HarperFlamingo Canada	|11676  |	8          |
| 4	  |0002005018 |	Clara Callan        |	Richard Bruce Wright | HarperFlamingo Canada  |41385  |	0          |

Tambahkan variable baru untuk jumlah peringkat yang di tetapkan. hapus dari pengguna yang memberi peringkat beberapa kali.

Table 2.penambahan fiture

|index| ISBN      |	Book-Title              |	Book-Author|	Publisher       |	User-ID|	Book-Rating|	Number of Book-Rating_x|	Number of Book-Rating_y|
|-----|-----------|-------------------------|------------|-----------       |--------|-------------|-------------------------|-------------------------|
|0    |	0399135782|	The Kitchen God's Wife	|Amy Tan     |	Putnam Pub Group|	8      |	0          |	33                     |	33                     |
|1    |	0399135782|	The Kitchen God's Wife	|Amy Tan     |	Putnam Pub Group|	11676  |	9          |	33                     |	33                     |
|2    |	0399135782|	The Kitchen God's Wife	|Amy Tan     |	Putnam Pub Group|	29526  |	9          |	33                     |	33                     |
|3    |	0399135782|	The Kitchen God's Wife	|Amy Tan     |	Putnam Pub Group|	36836  |	0          |	33                     |	33                     |
|4    |	0399135782|	The Kitchen God's Wife	|Amy Tan     |	Putnam Pub Group|	46398  |	9          |	33                     |	33                     |

setelah penambahan fiture maka dataset berubah menjadi 313546 baris dan 8 columns

### Menghapus data duplikat
Menghapus data duplikat dapat membantu meningkatkan performa model.
data duplikat yang dimiliki sebanyak : 1519 
data yang tersisa : 313546

### membuat visualisasi rating buku
gambar 3. Hasil rating buku

![image](https://github.com/sharung/Recomendasi_system/assets/76006507/d482a8d2-cb25-4cd4-80c3-eae14c94975b)

|Hasil rating| Jumlah rating|
|------------|--------------|
|30          | 5940         |
|32          | 5248         |
|31          | 4929         |
|33          | 4917         |
|34          | 4658         |
|            | ...          | 
|172         | 172          |
|163         | 163          |
|160         | 160          |
|141         | 141          |
|137         | 137          |

rating buku paling besar berada pada rating 30 dengan jumlah 5940


### Pembagian data uji dan latih
peda tahap ini dataset dibagi menjadi 2 yaitu data trainig(train) dan data validasi (test), pembagian ini bertujuan untuk mempersiapkan data rating yang akan digunakan dalam pelatihan dan pengujian model _collaborative filtering_. pembagian dataset dibagi dengan komposisi 80:20, 80% untuk data train dan 20% untuk data test

## Modeling
- Metode Colaborative Filtering

  Metode Colaborative filtering merupakan metode yang melakukan proses penyaringan item yang berdasarkan pengguna lain, dengan cara memberikan informasi kepada pengguna berdasarkan kemiripan karakteristik. Dalam pembuatanya saya menggunakan RecommenderNet, pada tahap ini model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding.
  
  - Data yang digunakan pada metode ini adalah data yang berupa nilai rating.
  - Top N Recommendation yang dihasilkan sebagai berikut.

### kelemahan _Collaborative-filtering_
- Kelemahan utama pada teknik ini yaitu sistem tidak dapat memberikan rekomendasi apabila belum adanya penilaian pada object yang di
rekomendasikan[1].
-  Collaborative-Filtering akan menghasilkan data yang kurang akurat ketika
penilaian pada satu data terlalu sedikit dan akan menjadi salah persepsi [1].
-  Teknik ini tidak memuat informasi / kegunaan dari barang yang
direkomendasikan[1].
### Kelebihan _Collaborative-filtering_
- memiliki kinerja yang baik
- akurasi yang baik

### Collaborative Filtering - teknik SVD
SVD (Singular Value Decomposition) merupakan teknik aljabar linear yang digunakan untuk mengurangi dimensi dan  menemukan relasi antar anggota di dalam matrix

Table 4. Hasil SVD
|Index| |ISBN|	Book-Title|	Book-Author|	prediction_rate |
|----|--|----|------------|------------|------------------|
|0|	|034545104X|	Flesh Tones: A Novel	|M. J. Rose	|0.30515|
|766517|	|0671665871|	GOING HOME	|Danielle Steel	|0.30515|
|766523|	|0671690582|	WHISPER OF DEATH : WHISPER OF DEATH	|Christopher Pike|	0.30515|
|766522|	|0671690094|	ANNE FRANK: DIARY OF A YOUNG GIRL	|Anne Frank|	0.30515|
|766521|	|067167630X|	BOOMER	|Charles Taylor	|0.30515|
|766520|	|067167465X|	BEST OF ENEMIES (NANCY DREW HARDY BOY SUPERMYS...	|Carolyn Keene|	0.30515|
|766519|	|0671673513|	The SILENCE OF THE LAMBS	|Thomas Harris|	0.30515|
|766518|	|0671670662|	WEB OF DREAMS (Casteel Saga (Paperback))	|V.C. Andrews	|0.30515|

# Evaluation
- Mean Squared Error (MSE)
MSE (Mean Squared Error) adalah salah satu metrik evaluasi yang umum digunakan dalam masalah regresi. Ini mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya dalam data.

MSE menghitung berapa rata-rata kesalahan kuadrat dalam prediksi. Semakin kecil nilai MSE, semakin baik kualitas model tersebut.

Rumus MSE:

MSE = (1/n) * Σ|i=1|^n (y_i - ŷ_i)^2

Di mana:
   
    n adalah jumlah sampel dalam data
    
    y_i adalah nilai aktual
    
    ŷ_i adalah nilai prediksi


- Root Mean Square Error (RMSE)
RMSE (Root Mean Squared Error) adalah metrik evaluasi yang umum digunakan dalam pemodelan regresi untuk mengukur sejauh mana selisih antara nilai prediksi dan nilai sebenarnya. RMSE menghitung akar kuadrat dari rata-rata kesalahan kuadrat antara nilai prediksi dan nilai sebenarnya.

Rumus RMSE dapat dituliskan sebagai berikut:

RMSE = sqrt(mean((nilai_pred - nilai_actual)^2))

di mana:
    
    n adalah jumlah sampel.
    
    Σ adalah tanda sigma yang menunjukkan penjumlahan.
    
    nilai_pred adalah nilai prediksi.
    
    nilai_actual adalah nilai sebenarnya.

Dengan penggunaan metrik tersebut, dapat dievaluasi seberapa baik model dalam memprediksi preferensi pengguna terhadap item secara akurat. Semakin rendah MAE dan RMSE maka itu menunjukkan model semakin akurat.

### Hasil 
Table 5. Hasil Evaluasi

|Evaluasi | Mean|
|---------|-----|
|RMSE     |0.798940619866297|
|MAE      |0.6949411091659635|

# Kesimpulan

Sistem rekomendasi buku sudah berhasil dibuat dengan metode collaborative filtering dengan teknik SDV, pemilihan ini dipilih karena sangat baik dari hasil model dan evaluasi yang menggunakan metode Mean Square error(MSE) dan Root Mean Square Error(RMSE), hasil evaluasi yang kecil dari RMSE mengartikan bahwa hasil rekomendasi sangat mendekati prediksi yang di berikan oleh model.

# Daftar Refrensi
[1] Islamiyah Mufidatul ,Subekti Puji ,Andini  Dwi Titania [Pemanfaatan Metode Item Based Collaborative Filtering Untuk 
Rekomendasi Wisata Di Kabupaten Malang](https://jurnal.stmikasia.ac.id/index.php/jitika/article/download/70/249/)

[2] Nur Hidayati, Arief Hermawan [K-Nearest Neighbor (K-NN) algorithm with Euclidean and Manhattan in classification of student graduation](https://journal.uny.ac.id/index.php/jeatech/article/viewFile/42777/pdf)

[3] Bayu, Sundawa Firdiansyah, Azhari Muh. [Implementasi Algoritma Brute Force Sebagai Mesin Pencari (Search Engine) Berbasis Web Pada Database](https://core.ac.uk/download/pdf/288088999.pdf)

