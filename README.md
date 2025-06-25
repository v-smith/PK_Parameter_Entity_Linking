
# PK_Parameter_Entity_Linker 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/v-smith/PK_Parameter_Entity_Linker/blob/master/LICENSE) ![version](https://img.shields.io/badge/version-0.1.0-blue) 


[**About the Project**](#about-the-project) | [**Dataset**](#dataset) | [**Getting Started**](#getting-started-) | [**Usage**](#usage) | [**Licence**](#lincence) | [**Citation**](#citation)

## About the Project

This repository contains custom pipelines and models to classify PK parameters mentions in sentences and tables from scientific publications against a PK ontology.

#### Project Structure

- The main code is found in the root of the repository (see Usage below for more information).

```
├── annotation guidelines # used by annotators for annotating data in this project
├── pk_el # code for data preprocessing, ontology preprocessing, tokenization, and entity linking approaches.
├── scripts  # scripts for model training and inference.
├── tests
├── .gitignore
├── LICENCE
├── README.md
├── requirements.txt
└── setup.py
```

#### Built With

[![Python v3.9](https://img.shields.io/badge/python-v3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)



## Dataset

The annotated PKEL corpora for sentences and tables are included in this repository. The PK ontology is also included in .csv format. The data is available under an MIT licence. The code assumes data is located in the `data` folder. 

## Getting Started 

#### Installation

To clone the repo:

`git clone https://github.com/PKPDAI/PK_Parameter_Entity_Linking`
    
To create a suitable environment:
- ```conda create --name PK_Parameter_Entity_Linking python==3.9```
- `conda activate PK_Parameter_Entity_Linking`
- `conda install pytorch torchvision torchaudio -c pytorch`
- `pip install -e .`

#### GPU Support

Using GPU is recommended. Single-GPU training has been tested with:
- `NVIDIA® GeForce RTX 30 series`
- `cuda 12.2`

## Usage

#### Tune the zero-shot bi-encoder

````bash
python scripts/linking/tuning/tune_biencoder_linker.py \
--match-threshold 0.80 \
--text-feature mention_with_window \
--table-feature text_with_tagged_mention \
--k 5 \
--category-constraint \
--include-pk_ontology-desc
````

#### Train the bi-encoder

````bash
python scripts/linking/tuning/train_biencoder.py \
--model-name intfloat/e5-small-v2 \
--text-feature mention_with_window \
--table-feature text_with_tagged_mention \
--use-hard-negatives \
--num-hard-negatives 1 \
--use-early-stopping \
--run-name my_biencoder_trial
````

####  Run prompt tuning 
```
python scripts/linking/tuning/tune_prompt_linker.py \
  --model-name gpt-4o-mini \
  --use-context \
  --subset-ontology \
  --use-examples \
  --n-runs 1
```


## License

The codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

[mit]: LICENCE

## Citation

```bibtex
tbc 
```

