# IndoNLU 
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlu/blob/master/LICENSE) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

*Baca README ini dalam [Bahasa Indonesia](README.id.md).*

<b>IndoNLU</b> is a collection of Natural Language Understanding (NLU) resources for Bahasa Indonesia with 6 kind of downstream tasks. We provide the code to reproduce the results and large pre-trained models (<b>IndoBERT</b> and <b>IndoBERT-lite</b>) trained with around 4 billion word corpus (<b>Indo4B</b>), more than 25 GB of text data. This project was initially started by a joint collaboration between universities and industry, such as Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, Gojek, and Prosa.AI.

## Research Paper
IndoNLU has been accepted by AACL-IJCNLP 2020 and you can find the details in our paper https://www.aclweb.org/anthology/2020.aacl-main.85.pdf.
If you are using any component on IndoNLU including Indo4B, FastText-Indo4B, or IndoBERT in your work, please cite the following paper:
```
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```

## How to contribute to IndoNLU?
Be sure to check the [contributing guidelines](https://github.com/indobenchmark/indonlu/blob/master/CONTRIBUTING.md) and contact the maintainers or open an issue to collect feedbacks before starting your PR.

# IndoNLG Downstream Task
Download and unzip the dataset from https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip

```
wget https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip
unzip downstream_task_datasets.zip
rm downstream_task_datasets.zip
```

## Indo4B Dataset
We provide the access to our large pretraining dataset.
- Indo4B-Plus Dataset Upscaled (~30 GB uncompressed, 9.4 GB compressed) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt.zip)

## IndoBART and IndoGPT2 Models
We provide IndoBART and IndoGPT2 Pretrained Language Model [[Link]](https://huggingface.co/indobenchmark)
- IndoBART [[Link]]()
- IndoGPT2 [[Linkg]]()

## Leaderboard
- Community Portal and Public Leaderboard [[Link]](https://www.indobenchmark.com/leaderboard.html)
- Submission Portal https://competitions.codalab.org/competitions/26537
