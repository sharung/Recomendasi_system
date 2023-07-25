# Laporan Proyek Machine Learning - ADAM PAMUNGKAS

## Project Overview

Perkembangan teknologi yang sangat pesat dibidang digital membuat manusia dapat dengam mudah mengakses segala sesuatu yang berada di internet salah satunya multimedia. perkembangan multimedia di dunia digital membuat manusia dapat mengakses beragam vidio, musik dan buku.

perkembangan dunia digital berpengaruh terhadap perkembangan alat manusia salah satunya adalah buku, buku pada era digital ini sangat mudah untuk dicari. sistem rekomendasi membantu untuk menawarkan buku yang mungkin akan cocok dengan kriterianya. dengan adanya sistem rekomendasi buku, pengguna dapat dengan mudah memilih buku baru atau buku yang relefan dengan pengguna tanpa menghabiskan waktu.
  
## Business Understanding
### Problem Statements
- Bagaimana cara mendapatkan rekomendasi buku yang cocok ?
### Goals
- Membuat sistem rekomendasi buku
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
  
User-ID: id pengguna sebagai penanda pengguna yang biasanya berupa angka
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
    
    rating_book.drop_duplicates(['User-ID', 'Book-Title'], inplace=True)
    rating_book.shape
    
data duplikat yang dimiliki sebanyak : 1519 
data yang tersisa : 313546

### membuat pivot table
pembuatan pivot table digunakan untuk mendapatkan nilai terdekat antara title buku dan rating buku.


## Modeling
- Metode Colaborative Filtering

  Metode Colaborative filtering merupakan metode yang melakukan proses penyaringan item yang berdasarkan pengguna lain, dengan cara memberikan informasi kepada pengguna berdasarkan kemiripan karakteristik. Dalam pembuatanya saya menggunakan RecommenderNet, pada tahap ini model menghitung skor kecocokan antara pengguna dan musik dengan teknik embedding.
  
  - Data yang digunakan pada metode ini adalah data yang berupa nilai, biasanya rating.
  - Top N Recommendation yang dihasilkan sebagai berikut.
 
Table 3. hasil recomendasi

|index|Judul                    |
|-----|-----                    |
|1    |'Fatal Terrain'          |
|2    |'Golden Cup'             |
|3    |'GEMINI CONTENDERS'      |
|4    |'Hidden Leaves (Debeers)'|
|5    |'Tall, Dark, and Deadly' |

adapun kelemahan pada _collaborative-filtering_
### kelemahan _Collaborative-filtering_
- Kelemahan utama pada teknik ini yaitu sistem tidak dapat memberikan rekomendasi apabila belum adanya penilaian pada object yang di
rekomendasikan[1].
-  Collaborative-Filtering akan menghasilkan data yang kurang akurat ketika
penilaian pada satu data terlalu sedikit dan akan menjadi salah persepsi [1].
-  Teknik ini tidak memuat informasi / kegunaan dari barang yang
direkomendasikan[1].
### Kelebihan
- memiliki kinerja yang baik
- akurasi yang baik

### Methhode NearestNeighbors dengan algoritma 
Algoritma Brute Force kNN menghitung jarak kuadrat dari setiap vektor fitur kueri ke setiap vektor fitur referensi dalam kumpulan data pelatihan. Kemudian, untuk setiap vektor fitur kueri, ia memilih objek dari set pelatihan yang paling dekat dengan vektor fitur kueri tersebut.
#### Kelebihan KNN 
- Mudah diterapkan.
- Mudah beradaptasi.
- Memiliki sedikit hyperparameter.
#### Kekurangan KNN. 
- Tidak berfungsi dengan baik pada dataset berukuran besar.
- Kurang cocok untuk dimensi tinggi.
- Perlu penskalaan fitur.
- Sensitif terhadap noise data, missing values dan outliers.

### Pelatihan menggunakan Brute Force
Selama pelatihan dengan pendekatan Brute Force, algoritme menyimpan semua vektor fitur dari kumpulan data pelatihan untuk menghitung jaraknya ke vektor fitur kueri.

# Evaluation
- Mean Squared Error (MSE)
MSE (Mean Squared Error) adalah salah satu metrik evaluasi yang umum digunakan dalam masalah regresi. Ini mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya dalam data


- Root Mean Square Error (RMSE)
RMSE (Root Mean Squared Error) adalah metrik evaluasi yang umum digunakan dalam pemodelan regresi untuk mengukur sejauh mana selisih antara nilai prediksi dan nilai sebenarnya. RMSE menghitung akar kuadrat dari rata-rata kesalahan kuadrat antara nilai prediksi dan nilai sebenarnya.

Nilai RMSE yang rendah dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai, Semakin kecil nilai RSME semakin dekat nilai yang di prediksi.

### Hasil 
|Evaluasi | Mean|
|---------|-----|
|RMSE     |3.740682932553402|
|MAE      |3.2001200612136103|

dapat disimpulkan bahwa Data yang dihasilkan sangat kecil yang berarti prediksi buku semakin dekat.

# Kesimpulan
Setelah melakukan beberapa kali pembersihan data, prediksi dapat berjalan dengan baik dikarenakan pemodelan menggunakan KNN sangat baik dari hasil model dan evaluasi yang menggunakan metode Mean Square error(MSE) dan Root Mean Square Error(RMSE), hasil evaluasi yang kecil dari RMSE mengartikan bahwa hasil rekomendasi sangat mendekati prediksi yang di berikan oleh model.

# Daftar Refrensi
[1] rendinusap [https://elib.unikom.ac.id/download.php?id=351950](https://elib.unikom.ac.id/download.php?id=351950)
