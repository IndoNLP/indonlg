# IndoNLG 
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlg/blob/master/LICENSE) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

*Baca README ini dalam [Bahasa Indonesia](README.id.md).*

<b>IndoNLG</b> is a collection of Natural Language Generation (NLG) resources for Bahasa Indonesia with 6 kind of downstream tasks. We provide the code to reproduce the results and large pre-trained models (<b>IndoBART</b> and <b>IndoGPT</b>) trained with around 4 billion word corpus (<b>Indo4B-Plus</b>), around ~25 GB of text data. This project was initially started by a joint collaboration between universities and industry, such as Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, DeepMind, Gojek, and Prosa.AI.

## Research Paper
IndoNLG has been accepted by EMNLP 2021 and you can find the details in our paper https://aclanthology.org/2021.emnlp-main.699.
If you are using any component on IndoNLG including Indo4B-Plus, IndoBART, or IndoGPT in your work, please cite the following paper:
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

### Example
- We provide example to load IndoBART model and fine-tune the model on Machine Translation task.
- Check our example on the following [Link](https://github.com/indobenchmark/indonlg/tree/master/examples)

## How to contribute to IndoNLG?
Be sure to check the [contributing guidelines](https://github.com/indobenchmark/indonlg/blob/master/CONTRIBUTING.md) and contact the maintainers or open an issue to collect feedbacks before starting your PR.

# IndoNLG Downstream Task
Download and unzip the dataset from https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip

```
wget https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip
unzip downstream_task_datasets.zip
rm downstream_task_datasets.zip
```

## Indo4B-Plus Dataset
We provide the access to our large pretraining dataset.
- Indo4B-Plus Dataset Upscaled (~25 GB uncompressed, 9.4 GB compressed) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/IndoNLG_ALL_new_dataset_preprocessed_uncased.txt.zip)

## IndoBART and IndoGPT Models
We provide IndoBART and IndoGPT Pretrained Language Model [[Link]](https://huggingface.co/indobenchmark)
- IndoBART [[Link]](https://huggingface.co/indobenchmark/indobart)
- IndoBART-v2 [[Link]](https://huggingface.co/indobenchmark/indobart-v2)
- IndoGPT [[Link]](https://huggingface.co/indobenchmark/indogpt)

## Indobenchmark Toolkit
We provide the toolkit to use the IndoNLGTokenizer in [[Link]](https://pypi.org/project/indobenchmark-toolkit/)
