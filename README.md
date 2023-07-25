# Laporan Proyek Machine Learning - ADAM PAMUNGKAS

## Project Overview

Perkembangan teknologi yang sangat pesat dibidang digital membuat manusia dapat dengam mudah mengakses segala sesuatu yang berada di internet salah satunya multimedia. perkembangan multimedia di dunia digital membuat manusia dapat mengakses beragam vidio, musik dan buku.

perkembangan dunia digital berpengaruh terhadap perkembangan alat manusia salah satunya adalah buku, buku pada era digital ini sangat mudah untuk dicari. sistem rekomendasi membantu untuk menawarkan buku yang mungkin akan cocok dengan kriterianya. dengan adanya sistem rekomendasi buku, pengguna dapat dengan mudah memilih buku baru atau buku yang relefan dengan pengguna tanpa menghabiskan waktu.
  
## Business Understanding
### Problem Statements
- Bagaimana cara membangun model untuk merekomendasikan buku yang cocok ?
### Goals
- Membuat sistem rekomendasi buku
### Solusi Approach
Solusi algoritma machine learning untuk sistem rekomendasi yaitu:

- Collaborative Filtering adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain.

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

- description of the book: recommending similar topics, could be solved by categorization to tags / keywords
- author details (these are mainly hypotheses for further exploration):
  - gender: men might not be interested in women topics and novels, but could be solved by categorization to tags / keywords
  - age: user might be more interested in books from authors with similar age due to same cultural/historical background, problematic with deceased authors
  - location: user may be interested in authors coming from similar location, i.e. Czech user interested in Czech author, or Polish user in Polish author
- languages, in which the book is available: we do not want to recommend books that are in the language not known to the user, because he/she would not be able to read it anyway (prefilter the language at the beginning)
- keywords: similarity of the books, content-based approach
- genre: recommending similar books based on genre, they have higher probability that the user will read them
- link between the genres: can help us with expanding the recommendation to other genres, that user might not have read yet but might be interested in (i.e. reader or historical novels can be interested in war literature even though he/she did not read anything like that yet)
- protagonist: we might relate more to the communication style of main characters with the same gender
- writing style: diary form, poetry, ich form, changing point of views, not categorized in genre but different people might be comfortable with different storytelling
- user behavior: this may contain data about how the users interacts with the books, what books he/she saves for later to read, or to buy, wish lists, favourite authors, clicks from the frontend
- parent book id: ISBN may not be appropriate to be used for recommendations, because unlike films and songs

  

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
data duplikat yang dimiliki sebanyak : 1,519 

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
- Kelemahan utama pada teknik ini yaitu sistem tidak dapat memberikan rekomendasi apabila belum adanya penilaian pada object yang di
rekomendasikan[1].
-  Collaborative-Filtering akan menghasilkan data yang kurang akurat ketika
penilaian pada satu data terlalu sedikit dan akan menjadi salah persepsi [1].
-  Teknik ini tidak memuat informasi / kegunaan dari barang yang
direkomendasikan[1].

# Daftar Refrensi
[1] rendinusap [https://elib.unikom.ac.id/download.php?id=351950](https://elib.unikom.ac.id/download.php?id=351950)
