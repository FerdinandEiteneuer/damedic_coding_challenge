# ICD Code Recommendation System 

## Installation

```bash
git clone git@github.com:FerdinandEiteneuer/damedic_coding_challenge.git
cd damedic_coding_challenge
python setup.py install
```

Content of `requirements.txt` is tensorflow. The python version used was 3.8.

## Usage

Before using the program, copy `train.csv` and `test.csv` into one folder, e.g. the data folder. This folder is input for the main skript `icd_recommender.py`.

```python
python damedic_coding_challenge/icd_recommender.py ./data/
```

The recommendations are found inside `./recommendations.csv`.

Please note that about 8GB of free memory are required.

## Creating the ICD recommendations

The approach to create the patient recommendations was inspired by [“Collaborative Denoising Autoencoders for Top-N Recommender Systems”](https://alicezheng.org/papers/wsdm16-cdae.pdf). Steps:


1. First, create corrupted training data by removing icd codes from patient cases at random.
2. Create an autoencoder neural network with one hidden layer. Input is the corrupted training data, targets are the original training samples with all icd codes. This way, the network learns to reproduce the full case.
3. After training, use the network to process the data from the test samples.
4. To get the recommendations, use the 5 highest values of the predictions, while excluding the icd codes that were already present in the test samples.

Since the random seeds are fixed, the training process should be reproducible and require about 25 Epochs.

