# Exploring Multimodal Models for Humor Recognition in Portuguese

This repository contains the experiments and results for the paper "Exploring Multimodal Models for Humor Recognition in Portuguese" accepted to the [16th International Conference on Computational Processing of Portuguese (PROPOR 2024)](https://propor2024.citius.gal/).

The main focus of this paper is to explore multimodal approaches in Humor Recognition by combining textual and explicit features. According to the architecture shown below.

```mermaid
flowchart BT
    text(Input text)
    transformer(Transformer)
    input(Input features)
    combination(Combination module)
    classification(Classification head)

    text-->transformer
    transformer-->combination;
    input--->combination;
    combination-->classification;
```

## Corpus - Puntuguese

The corpus is available on [GitHub](https://github.com/Superar/Puntuguese).

## How to run

This project is available as a python module to make installation and re-use easier.

Using Pipenv, just run:

```
pipenv install
```

If you prefer to use pip, run:

```
pip install -r requirements.txt
pip install .
```

## Repository organization

The repository is organized as follows:

```
multimodal-humor-recognition
├───results [Final prediction files and images used in the paper]
├───scripts [Jupyter notebooks of the experiments]
└───src [Multimodal models implementation]
```

## How to cite
```bibtex
@inproceedings{InacioGoncaloOliveira2024,
  title = {Exploring Multimodal Models for Humor Recognition in {{Portuguese}}},
  booktitle = {Proceedings of the 16th International Conference on Computational Processing of Portuguese},
  author = {In{\'a}cio, Marcio and Gon{\c c}alo Oliveira, Hugo},
  editor = {Gamallo, Pablo and Claro, Daniela and Teixeira, Ant{\'o}nio and Real, Livy and Garcia, Marcos and Oliveira, Hugo Gon{\c c}alo and Amaro, Raquel},
  year = {2024},
  month = mar,
  pages = {568--574},
  publisher = {Association for Computational Lingustics},
  address = {Santiago de Compostela, Galicia/Spain},
  url = {https://aclanthology.org/2024.propor-1.62},
  copyright = {All rights reserved}
}
```

