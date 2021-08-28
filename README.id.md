# IndoNLG 
![Ditunggu Pull Requestsnya](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![Lisensi Github](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlu/blob/master/LICENSE) [![Perjanjian Kontributor](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

*Read this README in [English](README.md).*

<b>IndoNLG</b> adalah sebuah koleksi sumber untuk riset dalam topik Natural Language Generation (NLG) untuk Bahasa Indonesia dengan 6 jenis downstream task. Kami menyediakan kode untuk mereproduksi hasil dan model besar yang sudah dilatih sebelumnya (<b>IndoBART</b> dan <b>IndoGPT2</b>) yang dilatih dengan kumpulan tulisan berisi sekitar 4 miliar kata dalam 3 bahasa: Indonesia, Sunda, dan Jawa (<b>Indo4B-Plus</b>) dan lebih dari 25 GB dalam ukuran data teks. Proyek ini awalnya dimulai dari kerjasama antara universitas dan industri, seperti Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, Gojek, Prosa.AI, dan DeepMind.

## Makalah Penelitian
IndoNLG telah diterima oleh EMNLP 2021 dan Anda dapat menemukan detailnya di paper kami https://arxiv.org/pdf/2104.08200.pdf.
Jika Anda menggunakan komponen apa pun di IndoNLG termasuk Indo4B-Plus, IndoBART, atau IndoGPT2 dalam pekerjaan Anda, harap kutip makalah berikut:

```
@misc{cahyawijaya2021indonlg,
      title={IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation}, 
      author={Samuel Cahyawijaya and Genta Indra Winata and Bryan Wilie and Karissa Vincentio and Xiaohong Li and Adhiguna Kuncoro and Sebastian Ruder and Zhi Yuan Lim and Syafri Bahar and Masayu Leylia Khodra and Ayu Purwarianti and Pascale Fung},
      year={2021},
      eprint={2104.08200},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Bagaimana cara untuk berkontribusi ke IndoNLU?
Pastikan anda mengecek [pedoman kontribusi](https://github.com/indobenchmark/indonlg/blob/master/CONTRIBUTING.md) dan hubungi pengelola atau buka issue untuk mengumpulkan umpan balik sebelum memulai PR Anda.

# IndoNLG Downstream Task
Download and unzip the dataset from https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip

```
wget https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip
unzip downstream_task_datasets.zip
rm downstream_task_datasets.zip
```

### Contoh
- Panduan untuk memuat model IndoBERT dan menyempurnakan model pada tugas Sequence Classification dan Sequence Tagging.
- Cek disini: [tautan](https://github.com/indobenchmark/indonlu/tree/master/examples)

### Susunan Pengiriman
Dimohon untuk memeriksa [tautan ini] (https://github.com/indobenchmark/indonlu/tree/master/submission_examples). Untuk setiap tugas, ada format yang berbeda. Setiap file pengiriman selalu dimulai dengan kolom `index` (id sampel pengujian mengikuti urutan set pengujian yang disamarkan).

Untuk pengiriman, pertama-tama Anda perlu mengganti nama prediksi Anda menjadi `pred.txt`, lalu membuat file menjadi zip. Setelah itu, Anda perlu mengizinkan sistem untuk menghitung hasilnya. Anda dapat dengan mudah memeriksa kemajuan anda di tab `hasil` Anda.

## Indo4B Dataset
Kami menyediakan akses ke kumpulan data pra-pelatihan kami yang besar.
- Indo4B-Plus Dataset Upscaled (~30 GB tidak dikompresi, 9.4 GB dikompresi) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt.zip)

## Model IndoBART and IndoGPT2 
Kami menyediakan Pretrained Language Model IndoBART dan IndoGPT2  [[Link]](https://huggingface.co/indobenchmark)
- IndoBART [[Link]]()
- IndoGPT2 [[Linkg]]()
