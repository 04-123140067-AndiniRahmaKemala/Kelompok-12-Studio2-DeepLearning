# Studio 2 — Klasifikasi Sampah · Kelompok 12

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap               | NIM               | Kontribusi utama |
| -- | -------------------------- | ----------------- | ---------------- |
| 1  | Alliyah Salsabilla | 123140014 | Kontribusi   |
| 2  | Andini Rahma Kemala | 123140067 | Kontribusi   |
| 3  | Zahwa Natasya Hamzah | 123140069 | Kontribusi  |

## Ringkasan Pendekatan

* **Arsitektur / backbone**: EfficientNet-B2 pretrained ImageNet menggunakan pustaka `timm`, dengan lapisan klasifikasi yang disesuaikan untuk menghasilkan prediksi lima kelas sampah.
* **Strategi pelatihan**: Transfer learning dan fine-tuning menggunakan optimizer AdamW dengan learning rate awal `3e-4`, batch size 32, dan rencana pelatihan 15 epoch. Eksperimen menggunakan learning rate scheduler dengan warm-up dan cosine decay, mixed-precision training, class weighting, serta label smoothing sebesar 0,1. Model terbaik dipilih berdasarkan Macro-F1 pada split validasi lokal.
* **Data tambahan**: Tidak menggunakan data tambahan dari luar dataset resmi Studio 2. Pelatihan menggunakan dataset yang disediakan untuk kompetisi. Dataset gambar tidak disertakan dalam repositori publik.
* **Ukuran input & preprocessing**: Gambar diproses sebagai RGB dengan ukuran input 288 × 288 piksel. Pada pelatihan, augmentasi yang direncanakan mencakup `RandomResizedCrop`, `RandomHorizontalFlip`, `TrivialAugmentWide`, dan `RandomErasing`, serta normalisasi menggunakan mean dan standard deviation ImageNet. Transformasi pada saat evaluasi mengikuti preprocessing yang sesuai dengan model terlatih tanpa augmentasi acak.

Model menghasilkan prediksi untuk lima kelas dengan urutan label berikut:

| Label | Kelas           |
| ----- | --------------- |
| 0     | `botol_plastik` |
| 1     | `gelas_plastik` |
| 2     | `kertas_kardus` |
| 3     | `saset_kemasan` |
| 4     | `styrofoam`     |

Metrik utama yang digunakan adalah **Macro-F1**, dengan akurasi sebagai metrik pendamping. Macro-F1 dipilih agar performa pada setiap kelas diperhitungkan secara seimbang.

## Hasil

| Putaran            | Commit          | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
| ------------------ | --------------- | ------------------------------- | ---------------------------- |
| Preview 1 (14 Okt) | `<Hash commit>` | `<Belum diisi>`                 | `<Belum diumumkan>`          |
| Preview 2 (21 Okt) | `<Hash commit>` | `<Belum diisi>`                 | `<Belum diumumkan>`          |
| Final (25 Okt)     | `<Hash commit>` | `<Belum diisi>`                 | `<Belum diumumkan>`          |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

* Eksperimen awal menggunakan EfficientNet-B2 pretrained ImageNet sebagai backbone klasifikasi.
* Pemilihan model dilakukan berdasarkan Macro-F1 pada split validasi lokal.
* Evaluasi lokal digunakan untuk membandingkan eksperimen, sedangkan skor yang diumumkan asisten berasal dari evaluasi pada validation set tersembunyi.
* Hasil, perubahan konfigurasi, dan temuan setiap eksperimen akan diperbarui setelah pengujian selesai.

## Cara Reproduksi

### 1. Persiapan lingkungan

Gunakan Python dengan versi yang kompatibel dengan versi PyTorch dan `timm` yang digunakan dalam eksperimen.

Jika dependensi disimpan di `submission/requirements.txt`, jalankan:

```bash
pip install -r submission/requirements.txt
```

Paket dasar yang digunakan mencakup PyTorch, torchvision, timm, NumPy, dan Pillow. Tambahkan dependensi lain yang benar-benar diperlukan oleh kode pelatihan ke berkas requirements.

### 2. Melatih model

Kode eksperimen saat ini disimpan dalam notebook `Studio2_Kompetisi_Sampah_v2.ipynb`. Jalankan sel notebook secara berurutan setelah menyesuaikan lokasi dataset dengan lingkungan yang digunakan.

Pastikan pelatihan menyimpan bobot model terbaik ke:

```text
submission/model.pth
```

Bobot harus sesuai dengan arsitektur dan konfigurasi preprocessing yang didefinisikan dalam `submission/model.py`.

### 3. Mengevaluasi submission

Siapkan split validasi lokal dengan struktur direktori yang sesuai dengan skrip evaluasi resmi, kemudian jalankan:

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Perintah tersebut menggunakan `evaluate.py` dari starter kit resmi. Pastikan argumen dan struktur data sesuai dengan versi skrip yang digunakan dalam kompetisi.

Sebelum melakukan commit untuk setiap putaran, periksa bahwa submission dapat dijalankan, prediksi menghasilkan logits dengan bentuk `[batch, 5]`, dan evaluasi seluruh validation set dapat diselesaikan dalam batas waktu yang ditetapkan.

## Struktur Repositori

```text
repo-kelompok/
├── README.md
├── evaluate.py
├── submission/
│   ├── model.py
│   ├── model.pth
│   └── requirements.txt
└── Studio2_Kompetisi_Sampah_v2.ipynb
```

* `submission/model.py`: menyediakan fungsi `get_model()` dan `get_transform()`.
* `submission/model.pth`: menyimpan `state_dict` dari model terbaik.
* `submission/requirements.txt`: mencantumkan dependensi tambahan yang dibutuhkan oleh submission.
* `evaluate.py`: skrip evaluasi resmi dari starter kit.
* `Studio2_Kompetisi_Sampah_v2.ipynb`: notebook eksperimen dan pelatihan model.

**Catatan penting:** `get_model()` hanya membangun arsitektur dan tidak mengunduh bobot atau memuat `model.pth` di dalam fungsi tersebut. Pemuatan bobot dilakukan oleh mekanisme evaluasi. Pastikan transformasi evaluasi menerima gambar PIL RGB dan mengembalikan tensor dengan bentuk `[3, H, W]`.
