# IndoNLG 
![Ditunggu Pull Requestsnya](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![Lisensi Github](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlu/blob/master/LICENSE) [![Perjanjian Kontributor](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

*Read this README in [English](README.md).*

<b>IndoNLG</b> adalah sebuah koleksi sumber untuk riset dalam topik Natural Language Generation (NLG) untuk Bahasa Indonesia dengan 6 jenis downstream task. Kami menyediakan kode untuk mereproduksi hasil dan model besar yang sudah dilatih sebelumnya (<b>IndoBART</b> dan <b>IndoGPT2</b>) yang dilatih dengan kumpulan tulisan berisi sekitar 4 miliar kata dalam 3 bahasa: Indonesia, Sunda, dan Jawa (<b>Indo4B-Plus</b>) dan lebih dari 25 GB dalam ukuran data teks. Proyek ini awalnya dimulai dari kerjasama antara universitas dan industri, seperti Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, Gojek, Prosa.AI, dan DeepMind.

## Makalah Penelitian
IndoNLG telah diterima oleh EMNLP 2021 dan Anda dapat menemukan detailnya di paper kami https://arxiv.org/pdf/2104.08200.pdf.
Jika Anda menggunakan komponen apa pun di IndoNLG termasuk Indo4B-Plus, IndoBART, atau IndoGPT2 dalam pekerjaan Anda, harap kutip makalah berikut:

```
@inproceedings{cahyawijaya-etal-2021-indonlg,
    title = "{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation",
    author = "Cahyawijaya, Samuel and Winata, Genta Indra and Wilie, Bryan and Vincentio, Karissa and Li, Xiaohong and Kuncoro, Adhiguna and Ruder, Sebastian and Lim, Zhi Yuan and Bahar, Syafri and Khodra, Masayu and Purwarianti, Ayu and Fung, Pascale",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing", month = nov, year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.699",
    pages = "8875--8898",
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
- Panduan untuk memuat model IndoBART dan menyempurnakan model pada tugas Machine Translation.
- Cek disini: [tautan](https://github.com/indobenchmark/indonlg/tree/master/examples)

### Susunan Pengiriman
Dimohon untuk memeriksa [tautan ini] (https://github.com/indobenchmark/indonlu/tree/master/submission_examples). Untuk setiap tugas, ada format yang berbeda. Setiap file pengiriman selalu dimulai dengan kolom `index` (id sampel pengujian mengikuti urutan set pengujian yang disamarkan).

Untuk pengiriman, pertama-tama Anda perlu mengganti nama prediksi Anda menjadi `pred.txt`, lalu membuat file menjadi zip. Setelah itu, Anda perlu mengizinkan sistem untuk menghitung hasilnya. Anda dapat dengan mudah memeriksa kemajuan anda di tab `hasil` Anda.

## Indo4B-Plus Dataset
Kami menyediakan akses ke kumpulan data pra-pelatihan kami yang besar.
- Indo4B-Plus Dataset Upscaled (~25 GB tidak dikompresi, 9.4 GB dikompresi) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt.zip).

## Model IndoBART and IndoGPT
Kami menyediakan Pretrained Language Model IndoBART dan IndoGPT [[Link]](https://huggingface.co/indobenchmark).
- IndoBART [[Link]](https://huggingface.co/indobenchmark/indobart)
- IndoBART-v2 [[Link]](https://huggingface.co/indobenchmark/indobart-v2)
- IndoGPT [[Link]](https://huggingface.co/indobenchmark/indogpt)

## Indobenchmark Toolkit
Kami menyediakan toolkit untuk menggunakan IndoNLGTokenizer di [[Link]](https://pypi.org/project/indobenchmark-toolkit/).