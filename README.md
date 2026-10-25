# Studio 2 — Klasifikasi Sampah · Kelompok 12

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|---|---|---|---|
| 1 | Alliyah Salsabilla | 123140014 | Pengembangan dan evaluasi model |
| 2 | Andini Rahma Kemala | 123140067 | Pengembangan dan evaluasi model |
| 3 | Zahwa Natasya Hamzah | 123140069 | Pengembangan dan evaluasi model |

## Ringkasan Pendekatan

Proyek ini menggunakan pendekatan transfer learning dan fine-tuning untuk melakukan klasifikasi citra sampah ke dalam lima kelas. Pengembangan model dilakukan melalui tiga putaran, yaitu Preview 1, Preview 2, dan Final.

### Preview 1 — ResNet-18

- **Arsitektur / backbone:** ResNet-18 pretrained ImageNet dengan lapisan klasifikasi yang disesuaikan untuk memprediksi lima kelas sampah.
- **Strategi pelatihan:** Fine-tuning menggunakan optimizer AdamW, learning rate awal `3e-4`, batch size 32, dan 8 epoch. Pelatihan menggunakan weight decay `1e-4`, cosine annealing learning rate scheduler, serta Cross Entropy Loss dengan label smoothing `0.1`. Model terbaik disimpan berdasarkan akurasi validasi lokal.
- **Ukuran input dan preprocessing:** Gambar RGB berukuran 224 × 224 piksel. Augmentasi training mencakup `RandomResizedCrop` dengan skala 0.6–1.0, `RandomHorizontalFlip`, dan `ColorJitter`. Gambar dinormalisasi menggunakan mean dan standard deviation ImageNet. Validasi menggunakan `Resize(256)` dan `CenterCrop(224)` tanpa augmentasi acak.

### Preview 2 — EfficientNet-B2

- **Arsitektur / backbone:** EfficientNet-B2 pretrained ImageNet dengan lapisan klasifikasi yang disesuaikan untuk lima kelas sampah.
- **Strategi pelatihan:** Fine-tuning selama 15 epoch.
- **Ukuran input:** Gambar RGB berukuran 288 × 288 piksel.
- **Evaluasi:** Menggunakan Macro-F1 dan akurasi pada validation set lokal.

### Final — ConvNeXt-Tiny

- **Arsitektur / backbone:** ConvNeXt-Tiny pretrained ImageNet dengan classifier yang disesuaikan untuk memprediksi lima kelas sampah.
- **Strategi pelatihan:** Fine-tuning menggunakan optimizer AdamW, learning rate awal `1e-4`, weight decay `1e-4`, batch size 16, dan maksimum 15 epoch.
- **Pemilihan model terbaik:** Berdasarkan peningkatan Macro-F1 pada validation set lokal.
- **Learning rate scheduler:** `ReduceLROnPlateau` berdasarkan Macro-F1 validation.
- **Early stopping:** Diterapkan apabila Macro-F1 tidak meningkat selama lima epoch berturut-turut.
- **Mixed precision:** Menggunakan Automatic Mixed Precision (AMP) untuk meningkatkan efisiensi pelatihan pada GPU.
- **Ukuran input:** Gambar RGB berukuran 224 × 224 piksel.
- **Pembagian data:** Stratified split dengan 85% data training dan 15% data validation menggunakan random seed `42`.
- **Jumlah data:** 2.753 gambar, terdiri atas 2.340 gambar training dan 413 gambar validation.
- **Penyimpanan model:** Checkpoint terbaik disimpan dalam format `state_dict` PyTorch dan diekspor sebagai `submission/model.pth`.
- **Evaluasi:** Menggunakan Macro-F1 sebagai metrik utama dan akurasi sebagai metrik pendamping.

Urutan label model:

| Label | Kelas |
|---|---|
| 0 | `botol_plastik` |
| 1 | `gelas_plastik` |
| 2 | `kertas_kardus` |
| 3 | `saset_kemasan` |
| 4 | `styrofoam` |

Macro-F1 digunakan sebagai metrik utama evaluasi, dengan akurasi sebagai metrik pendamping.

## Hasil

| Putaran | Commit model | Macro-F1 (validasi lokal) | Macro-F1 (validation set tersembunyi) |
|---|---|---:|---|
| Preview 1 (14 Okt 2026) | `aec291b` | 0.9294 | Belum diumumkan |
| Preview 2 (21 Okt 2026) | `54b75ec` | 0.9524 | Belum diumumkan |
| Final (25 Okt 2026) | `fc6e48d` | 0.9694 | Belum diumumkan |

### Hasil evaluasi lokal Preview 1:
- Akurasi: `0.9317`
- Macro-F1: `0.9294`
- Jumlah gambar validasi lokal: 410
- Jumlah gambar training: 2.343

F1-score per kelas:
- `botol_plastik`: 0.9404
- `gelas_plastik`: 0.9367
- `kertas_kardus`: 0.9020
- `saset_kemasan`: 0.9261
- `styrofoam`: 0.9417

Validasi lokal menggunakan split sebesar 15% dari dataset training resmi. Hasil ini tidak mewakili skor validation set tersembunyi yang digunakan oleh tim pengajar.

Catatan eksperimen:
- Baseline awal menggunakan ResNet-18 pretrained ImageNet.
- Akurasi validasi terbaik selama training adalah 0.9317 pada epoch ke-5.
- Macro-F1 dan akurasi dihitung kembali menggunakan skrip evaluasi resmi pada validation set lokal.
- Eksperimen lanjutan dapat dilakukan pada putaran berikutnya untuk meningkatkan generalisasi model.

### Hasil evaluasi lokal Preview 2:

- Akurasi: `0.9561`
- Macro-F1: `0.9524`
- Jumlah gambar validasi lokal: 410
- Jumlah gambar training: 2.346

F1-score per kelas:

- `botol_plastik`: 0.9630
- `gelas_plastik`: 0.9615
- `kertas_kardus`: 0.9091
- `saset_kemasan`: 0.9697
- `styrofoam`: 0.9588

Validasi lokal menggunakan split sebesar 15% dari dataset training resmi. Hasil ini tidak mewakili skor validation set tersembunyi yang digunakan oleh tim pengajar.

Catatan eksperimen:

- Model menggunakan EfficientNet-B2 pretrained.
- Ukuran input gambar: `288 × 288` piksel.
- Training dilakukan selama 15 epoch.
- Macro-F1 dan akurasi dihitung menggunakan skrip evaluasi `evaluate.py` pada validation set lokal.
- F1-score tertinggi diperoleh kelas `saset_kemasan`, yaitu 0.9697, sedangkan terendah diperoleh kelas `kertas_kardus`, yaitu 0.9091.
- Dibandingkan Preview 1, akurasi meningkat dari 0.9317 menjadi 0.9561 dan Macro-F1 meningkat dari 0.9294 menjadi 0.9524.
- Eksperimen lanjutan dapat dilakukan untuk meningkatkan generalisasi model, khususnya pada kelas `kertas_kardus`.

### Hasil Evaluasi Lokal Final:

Model final menggunakan arsitektur ConvNeXt-Tiny pretrained ImageNet yang telah disesuaikan untuk klasifikasi lima kelas sampah.

Hasil evaluasi lokal:

- Akurasi: `0.9709` (97.09%)
- Macro-F1: `0.9694` (96.94%)
- Jumlah gambar validasi lokal: 413
- Jumlah gambar training: 2.340
- Total dataset: 2.753 gambar

#### F1-Score per Kelas

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `botol_plastik` | 0.9620 | 0.9744 | 0.9682 | 78 |
| `gelas_plastik` | 0.9383 | 0.9744 | 0.9560 | 78 |
| `kertas_kardus` | 0.9811 | 0.9455 | 0.9630 | 55 |
| `saset_kemasan` | 0.9898 | 0.9604 | 0.9749 | 101 |
| `styrofoam` | 0.9804 | 0.9901 | 0.9852 | 101 |

#### Confusion Matrix

| Kelas Aktual / Prediksi | Botol Plastik | Gelas Plastik | Kertas Kardus | Saset Kemasan | Styrofoam |
|---|---:|---:|---:|---:|---:|
| `botol_plastik` | 76 | 2 | 0 | 0 | 0 |
| `gelas_plastik` | 1 | 76 | 0 | 0 | 1 |
| `kertas_kardus` | 0 | 3 | 52 | 0 | 0 |
| `saset_kemasan` | 2 | 0 | 1 | 97 | 1 |
| `styrofoam` | 0 | 0 | 0 | 1 | 100 |

#### Catatan Eksperimen Final

- Model menggunakan ConvNeXt-Tiny pretrained ImageNet.
- Ukuran input gambar adalah `224 × 224` piksel.
- Batch size yang digunakan adalah `16`.
- Maksimum pelatihan adalah 15 epoch dengan mekanisme early stopping.
- Checkpoint terbaik dipilih berdasarkan Macro-F1 validation lokal.
- Pengujian dilakukan menggunakan skrip evaluasi resmi `evaluate.py` dari starter kit Studio 2.
- Model berhasil dimuat dan dijalankan menggunakan mekanisme submission resmi.
- Macro-F1 lokal yang diperoleh adalah 0.9694 dan akurasi lokal sebesar 0.9709.
- F1-score tertinggi diperoleh kelas `styrofoam`, yaitu 0.9852.
- F1-score terendah diperoleh kelas `gelas_plastik`, yaitu 0.9560.
- Hasil evaluasi lokal belum tentu sama dengan hasil validation set tersembunyi yang digunakan oleh tim pengajar.


## Perbandingan Hasil Evaluasi Preview 1, Preview 2, dan Final

| Parameter | Preview 1 | Preview 2 | Final |
|---|---|---|---|
| Arsitektur model | ResNet-18 | EfficientNet-B2 | ConvNeXt-Tiny |
| Ukuran input | 224 × 224 | 288 × 288 | 224 × 224 |
| Akurasi | 0.9317 (93.17%) | 0.9561 (95.61%) | **0.9709 (97.09%)** |
| Macro-F1 | 0.9294 (92.94%) | 0.9524 (95.24%) | **0.9694 (96.94%)** |
| Jumlah gambar training | 2.343 | 2.346 | 2.340 |
| Jumlah gambar validasi lokal | 410 | 410 | 413 |

### Perbandingan F1-Score per Kelas

| Kelas | Preview 1 | Preview 2 | Final |
|---|---:|---:|---:|
| `botol_plastik` | 0.9404 | 0.9630 | **0.9682** |
| `gelas_plastik` | 0.9367 | 0.9615 | 0.9560 |
| `kertas_kardus` | 0.9020 | 0.9091 | **0.9630** |
| `saset_kemasan` | 0.9261 | 0.9697 | **0.9749** |
| `styrofoam` | 0.9417 | 0.9588 | **0.9852** |

### Analisis Perbandingan

Berdasarkan hasil evaluasi lokal, model ConvNeXt-Tiny memperoleh Macro-F1 tertinggi dibandingkan kedua eksperimen sebelumnya.

Peningkatan Macro-F1 yang tercatat:

- Preview 1 ke Preview 2: `+0.0230`
- Preview 2 ke Final: `+0.0170`
- Preview 1 ke Final: `+0.0400`

Peningkatan akurasi yang tercatat:

- Preview 1 ke Preview 2: `+0.0244`
- Preview 2 ke Final: `+0.0148`
- Preview 1 ke Final: `+0.0392`

## Cara Reproduksi

### 1. Persiapan lingkungan

Instal dependensi dari root repositori:

```bash
pip install -r requirements.txt
```

Dataset harus diperoleh dari Google Drive resmi yang disediakan tim pengajar. Jangan mengunggah ulang dataset ke repositori publik.

Pastikan struktur dataset training terdiri dari folder lima kelas berikut:

```text
train/
├── botol_plastik/
├── gelas_plastik/
├── kertas_kardus/
├── saset_kemasan/
└── styrofoam/
```

### 2. Melatih model

Jalankan dari root repositori:

```bash
python train_baseline.py --data /path/to/train --epochs 8 --batch-size 32 --lr 3e-4 --seed 42
```

Ganti `/path/to/train` dengan lokasi folder `train/` pada lingkungan yang digunakan.

Skrip menghasilkan split validasi lokal `my_val/`, split training `my_train/`, dan menyimpan bobot model terbaik di `submission/model.pth`.

### Melatih Model Final — ConvNeXt-Tiny

Eksperimen final menggunakan ConvNeXt-Tiny dan dijalankan melalui notebook Google Colab.

Konfigurasi utama:

| Parameter | Nilai |
|---|---|
| Model | ConvNeXt-Tiny |
| Pretrained weights | ImageNet |
| Image size | 224 × 224 |
| Batch size | 16 |
| Maksimum epoch | 15 |
| Learning rate awal | 0.0001 |
| Weight decay | 0.0001 |
| Random seed | 42 |
| Validation split | 15% |
| Early stopping patience | 5 |
| Model selection | Best validation Macro-F1 |

Tahapan eksperimen final:

1. Memuat dataset resmi Studio 2.
2. Membagi dataset menjadi training dan validation secara stratified.
3. Menerapkan preprocessing dan augmentasi gambar.
4. Memuat ConvNeXt-Tiny pretrained ImageNet.
5. Mengganti classifier agar menghasilkan lima kelas.
6. Melakukan fine-tuning model.
7. Mengevaluasi model pada setiap epoch.
8. Menyimpan checkpoint dengan Macro-F1 validation terbaik.
9. Mengekspor bobot terbaik ke `submission/model.pth`.

Skrip `train_baseline.py` yang ada di repositori merupakan skrip baseline ResNet-18, bukan skrip training ConvNeXt-Tiny.

### 3. Mengevaluasi submission

Jalankan evaluasi pada split validasi lokal:

```bash
python evaluate.py --submission ./submission --data ./my_val --out ./results/preview1_val_result.json
```

Model submission harus menyediakan fungsi `get_model()` dan `get_transform()` di `submission/model.py`. Fungsi `get_model()` hanya membangun arsitektur tanpa mengunduh bobot atau memuat `model.pth`. Proses evaluasi akan memuat `state_dict` dari `submission/model.pth`.

Output model harus berupa logits dengan bentuk `[batch, 5]`, dengan urutan kelas sesuai tabel label.

## Struktur Repositori

```text
Kelompok-12-Studio2-DeepLearning/
├── ConvNeXt/
│   ├── convnext_tiny_history.json
│   └── official_eval_local.json
│
├── results/
│   ├── preview1_val_result.json
│   ├── preview2_train_log.json
│   └── preview2_val_result.json
│
├── submission/
│   ├── model.pth
│   ├── model.py
│   └── requirements.txt
│
├── .gitignore
├── README.md
├── evaluate.py
├── requirements.txt
├── train.py
└── train_baseline.py
```

- `README.md`: Dokumentasi eksperimen Preview 1, Preview 2, dan Final.
- `.gitignore`: Mengatur file yang tidak dilacak Git.
- `requirements.txt`: Dependensi yang diperlukan untuk menjalankan kode.
- `train_baseline.py`: Skrip pelatihan baseline ResNet-18.
- `train.py`: Skrip training EfficientNet-B2.
- `evaluate.py`: Skrip evaluasi resmi dari starter kit Studio 2.
- `ConvNeXt/`: Menyimpan riwayat training dan hasil evaluasi lokal ConvNeXt-Tiny.
- `results/`: Menyimpan hasil eksperimen Preview 1 dan Preview 2.
- `submission/model.py`: Definisi arsitektur model dan transformasi evaluasi.
- `submission/model.pth`: State dictionary model yang digunakan untuk submission.
- `submission/requirements.txt`: Dependensi khusus submission.
