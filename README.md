# Studio 2 — Klasifikasi Sampah · Kelompok 12

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|---|---|---|---|
| 1 | Alliyah Salsabilla | 123140014 | kontribusi |
| 2 | Andini Rahma Kemala | 123140067 | kontribusi |
| 3 | Zahwa Natasya Hamzah | 123140069 | kontribusi |

## Ringkasan Pendekatan

- **Arsitektur / backbone:** ResNet-18 pretrained ImageNet dengan lapisan klasifikasi yang disesuaikan untuk memprediksi lima kelas sampah.
- **Strategi pelatihan:** Fine-tuning menggunakan optimizer AdamW, learning rate awal `3e-4`, batch size 32, dan 8 epoch. Pelatihan menggunakan weight decay `1e-4`, cosine annealing learning rate scheduler, serta Cross Entropy Loss dengan label smoothing `0.1`. Model terbaik disimpan berdasarkan akurasi validasi lokal.
- **Data tambahan:** Tidak menggunakan data tambahan dari luar dataset resmi Studio 2. Dataset tidak disertakan dalam repositori publik.
- **Ukuran input dan preprocessing:** Gambar RGB dengan ukuran input 224 × 224 piksel. Augmentasi training mencakup `RandomResizedCrop` dengan skala 0.6–1.0, `RandomHorizontalFlip`, dan `ColorJitter`. Gambar dinormalisasi menggunakan mean dan standard deviation ImageNet. Validasi menggunakan `Resize(256)` dan `CenterCrop(224)` tanpa augmentasi acak.

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
| Preview 2 (21 Okt 2026) | Belum tersedia | 0.9524 | Belum diumumkan |
| Final (25 Okt 2026) | Belum tersedia | Belum diisi | Belum diumumkan |

Hasil evaluasi lokal Preview 1:
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

Hasil evaluasi lokal Preview 2:

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

## Perbandingan Hasil Evaluasi Preview 1 dan Preview 2

| Parameter | Preview 1 | Preview 2 | Perubahan |
|---|---:|---:|---:|
| Arsitektur model | ResNet-18 | EfficientNet-B2 | Diperbarui |
| Akurasi | 0.9317 (93.17%) | 0.9561 (95.61%) | +0.0244 |
| Macro-F1 | 0.9294 (92.94%) | 0.9524 (95.24%) | +0.0230 |
| Jumlah gambar training | 2.343 | 2.346 | +3 |
| Jumlah gambar validasi lokal | 410 | 410 | Tetap |

### Perbandingan F1-Score per Kelas

| Kelas | Preview 1 | Preview 2 | Perubahan |
|---|---:|---:|---:|
| `botol_plastik` | 0.9404 | 0.9630 | +0.0226 |
| `gelas_plastik` | 0.9367 | 0.9615 | +0.0248 |
| `kertas_kardus` | 0.9020 | 0.9091 | +0.0071 |
| `saset_kemasan` | 0.9261 | 0.9697 | +0.0436 |
| `styrofoam` | 0.9417 | 0.9588 | +0.0171 |

Catatan eksperimen:

- Model menggunakan EfficientNet-B2 pretrained.
- Ukuran input gambar: `288 × 288` piksel.
- Training dilakukan selama 15 epoch.
- Macro-F1 dan akurasi dihitung menggunakan skrip evaluasi `evaluate.py` pada validation set lokal.
- F1-score tertinggi diperoleh kelas `saset_kemasan`, yaitu 0.9697, sedangkan terendah diperoleh kelas `kertas_kardus`, yaitu 0.9091.
- Dibandingkan Preview 1, akurasi meningkat dari 0.9317 menjadi 0.9561 dan Macro-F1 meningkat dari 0.9294 menjadi 0.9524.
- Eksperimen lanjutan dapat dilakukan untuk meningkatkan generalisasi model, khususnya pada kelas `kertas_kardus`.


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

### 3. Mengevaluasi submission

Jalankan evaluasi pada split validasi lokal:

```bash
python evaluate.py --submission ./submission --data ./my_val --out ./results/preview1_val_result.json
```

Model submission harus menyediakan fungsi `get_model()` dan `get_transform()` di `submission/model.py`. Fungsi `get_model()` hanya membangun arsitektur tanpa mengunduh bobot atau memuat `model.pth`. Proses evaluasi akan memuat `state_dict` dari `submission/model.pth`.

Output model harus berupa logits dengan bentuk `[batch, 5]`, dengan urutan kelas sesuai tabel label.

## Struktur Repositori

```text
repo-kelompok/
├── README.md
├── .gitignore
├── requirements.txt
├── train_baseline.py
├── evaluate.py
├── results/
│   └── preview1_val_result.json
└── submission/
    ├── model.py
    └── model.pth
```

- `train_baseline.py`: skrip pelatihan baseline ResNet-18.
- `evaluate.py`: skrip evaluasi resmi dari starter kit.
- `submission/model.py`: definisi arsitektur dan transformasi evaluasi.
- `submission/model.pth`: `state_dict` model terbaik.
- `requirements.txt`: dependensi yang diperlukan untuk menjalankan kode.

Catatan: hasil evaluasi lokal berbeda dari skor validation set tersembunyi yang dihitung oleh tim pengajar.
