# AureaVoice Backend


## Ringkasan Proyek

AureaVoice Backend adalah sebuah layanan API yang dirancang untuk menganalisis dan mengklasifikasikan aksen dalam rekaman suara. Proyek ini bertujuan untuk menyediakan alat bantu bagi para pembelajar bahasa Inggris, khususnya untuk mengidentifikasi apakah aksen mereka mendekati aksen penutur asli Amerika Serikat (US accent). Aplikasi ini memecahkan masalah kurangnya umpan balik instan tentang pelafalan dan aksen bagi pembelajar mandiri.


## ⚙️ Komponen Utama

1.  **Client** (misalnya, aplikasi frontend) melakukan `POST request` ke endpoint `/classify-us-accent` pada **Backend API** dengan melampirkan file audio.
2.  **Backend API** menerima file audio dan menyimpannya sementara di server.
3.  File audio tersebut kemudian diteruskan ke **Model Klasifikasi Aksen** untuk dianalisis.
4.  Model memproses audio dan menghasilkan probabilitas untuk setiap aksen yang dikenali. API secara spesifik akan mengekstrak skor kepercayaan (confidence score) untuk aksen Amerika Serikat ('us').
5.  **Backend API** mengirimkan kembali respons JSON ke client yang berisi skor kepercayaan aksen US, misalnya: `{"us_confidence": 85.5}`.
6.  File audio sementara yang tadi disimpan akan dihapus dari server untuk menjaga privasi dan efisiensi penyimpanan.


## 🔬 Detail Alur Kerja Teknis
Bagaimana sebuah file audio bisa diubah menjadi skor kepercayaan (confidence score)? Proses ini melibatkan beberapa teknologi kunci yang bekerja sama.


*   **Torchaudio (Ahli Audio):** Komponen dari PyTorch ini bertugas membaca file audio `.wav` dan mengubahnya dari gelombang suara mentah menjadi format numerik (Tensor) yang dapat diproses oleh model. Ia juga melakukan normalisasi penting seperti menyeragamkan *sample rate* dan mengekstraksi fitur audio (Spectrogram/MFCCs) yang menjadi input bagi model.

*   **PyTorch (Otak Pemrosesan):** Sebagai *neural network engine*, PyTorch mengambil data numerik dari Torchaudio dan menjalankannya melalui model `Ecapa-TDNN`. Model ini menghasilkan *embedding*—representasi matematis dari karakteristik unik suara—yang kemudian dipetakan ke setiap kemungkinan aksen untuk menghasilkan skor mentah (*logits*).
*   **SpeechBrain (Manajer Proyek):** Framework ini mengorkestrasi seluruh alur kerja. Ia menyediakan model `EncoderClassifier` yang sudah terlatih dan menyederhanakan proses, mulai dari memanggil `torchaudio` untuk pra-pemrosesan hingga menjalankan inferensi dengan `PyTorch` dan mengelola outputnya.
*   **Python/NumPy (Interpreter Hasil):** Skor mentah (*logits*) dari PyTorch diubah menjadi probabilitas (0-1) menggunakan fungsi Softmax. Kode Python kemudian mencari probabilitas yang spesifik untuk aksen 'us', mengubahnya menjadi format persentase, dan menyajikannya dalam output JSON yang rapi.

Secara ringkas, alurnya adalah:
**File Audio → `torchaudio` (Konversi ke Tensor) → `PyTorch` (Analisis & Hasilkan Skor Mentah) → `Softmax` (Konversi ke Probabilitas) → `Python` (Format menjadi JSON).**

## Cuplikan Kode Penting
1.  **Inisialisasi API dan Model Klasifikasi**

    Cuplikan ini menunjukkan bagaimana server FastAPI diinisialisasi bersama dengan model SpeechBrain. Model dimuat saat aplikasi pertama kali dijalankan untuk memastikan respons yang cepat saat request masuk.

    ```python
    from fastapi import FastAPI
    from speechbrain.pretrained import EncoderClassifier

    app = FastAPI()


    # Memuat model klasifikasi aksen saat startup
    classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa"
    )

    ```

2.  **Endpoint Klasifikasi Aksen**
    Kode ini mendefinisikan endpoint `/classify-us-accent` yang menangani logika utama: menerima file audio, memanggil model untuk klasifikasi, dan mengembalikan hasilnya.

    ```python
    @app.post("/classify-us-accent")

    async def classify_us_accent(file: UploadFile = File(...)):
        # Simpan file audio yang diunggah untuk sementara

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            # Lakukan klasifikasi pada file audio

            out_prob, score, index, text_lab = classifier.classify_file(tmp_path)
            
            # Cari indeks untuk aksen 'us' dan ambil skor kepercayaannya

            us_index = # ... (logika untuk menemukan indeks 'us')
            us_confidence = float(out_prob[0][us_index]) * 100

            

            return {"us_confidence": round(us_confidence, 1)}
        finally:
            # Hapus file sementara setelah selesai
            os.remove(tmp_path)
    ```

## 🆘 Dukungan

Untuk masalah dan pertanyaan:
- Periksa bagian pemecahan masalah di atas

- Tinjau dokumentasi FastAPI dan SpeechBrain
- Buka isu di repositori

## 📚 Referensi

### Dokumentasi
- [Dokumentasi SpeechBrain](https://speechbrain.github.io/)
- [Dokumentasi FastAPI](https://fastapi.tiangolo.com/)
- [Dataset CommonAccent](https://huggingface.co/datasets/Jzuluaga/commonaccent)

### Makalah Penelitian
Jika Anda menggunakan karya ini dalam penelitian Anda, harap kutip makalah berikut:

**Dataset CommonAccent:**
```bibtex
@article{zuluaga2023commonaccent,
  title={CommonAccent: Exploring Large Acoustic Pretrained Models for Accent Classification Based on Common Voice},

  author={Zuluaga-Gomez, Juan and Ahmed, Sara and Visockas, Danielius and Subakan, Cem},
  journal={Interspeech 2023},
  url={https://arxiv.org/abs/2305.18283},
  year={2023}
}
```

**Arsitektur ECAPA-TDNN:**
```bibtex

@inproceedings{desplanques2020ecapa,
  author    = {Brecht Desplanques and
               Jenthe Thienpondt and
               Kris Demuynck},
  editor    = {Helen Meng and
               Bo Xu and
               Thomas Fang Zheng},
  title     = {{ECAPA-TDNN:} Emphasized Channel Attention, Propagation and Aggregation
               in {TDNN} Based Speaker Verification},
  booktitle = {Interspeech 2020},
  pages     = {3830--3834},
  publisher = {{ISCA}},
  year      = {2020},
}}
```

**Wav2Vec2 XLSR:**
```bibtex
@article{conneau2020unsupervised,
  title={Unsupervised cross-lingual representation learning for speech recognition},
  author={Conneau, Alexis and Baevski, Alexei and Collobert, Ronan and Mohamed, Abdelrahman and Auli, Michael},
  journal={arXiv preprint arXiv:2006.13979},
  year={2020}
}
```

**Toolkit SpeechBrain:**
```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}}
```
