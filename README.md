# atmt code base
Materials for "Advanced Techniques of Machine Translation".
Please refer to the assignment sheet for instructions on how to use the toolkit.

The toolkit is based on [this implementation](https://github.com/demelin/nmt_toolkit).

# Environment Setup

In case you prefer to run the code locally, we suggest creating a Python environment to prevent library clashes with future projects, using either Conda or virtualenv (Conda is suggested). For other options, see [supplementary material](https://neat-tortellini-10f.notion.site/ATMT-Autumn-2023-Assignment-1-Setup-Instructions-96d8444a7d7146139a5b76a86a559f5f?pvs=4)

### conda

```
# ensure that you have conda (or miniconda) installed (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and that it is activated

# create clean environment
conda create --name atmt311 python=3.11

# activate the environment
conda activate atmt311

# intall required packages
conda install pytorch=2.0.1 numpy tqdm sacrebleu
```

### virtualenv

```
# ensure that you have python > 3.6 downloaded and installed (https://www.python.org/downloads/)

# install virtualenv
pip install virtualenv  # for both powershell and WSL

# create a virtual environment named "atmt311"
virtualenv --python=python3.11 atmt311  # on WSL terminal
python -m venv atmt311    # on powershell

# launch the newly created environment
source atmt311/bin/activate
.\atmt311\Scripts\Activate.ps1   # on powershell


# intall required packages
pip install torch==2.0.1 numpy tqdm sacrebleu   # for both powershell and WSL
```

<!-- # Data Preprocessing

```
# normalise, tokenize and truecase data
bash scripts/extract_splits.sh ../infopankki_raw data/en-sv/infopankki/raw

# binarize data for model training
bash scripts/run_preprocessing.sh data/en-sv/infopankki/raw/
``` -->

# Training a model

```
python train.py \
    --data path/to/prepared/data \
    --source-lang en \
    --target-lang sv \
    --save-dir path/to/model/checkpoints \
    --train-on-tiny # for testing purposes only
```

Notes:
- `path/to/prepared/data` and `path/to/model/checkpoints`
  are placholders, not true paths. Replace these arguments with the correct paths
  for your system.
- only use `--train-on-tiny` for testing. This will train a
dummy model on the `tiny_train` split.
- add the `--cuda` flag if you want to train on a GPU, e.g. using Google Colab

# Evaluating a trained model

Run inference on test set
```
python translate.py \
    --data path/to/prepared/data \
    --dicts path/to/prepared/data \
    --checkpoint-path path/to/model/checkpoint/file/for/loading \
    --output path/to/output/file/model/translations
```

Postprocess model translations
```
bash scripts/postprocess.sh path/to/output/file/model/translations path/to/postprocessed/model/translations/file en
```

Score with SacreBLEU
```
cat path/to/postprocessed/model/translations/file | sacrebleu path/to/raw/target/test/file
```

# Assignments

Assignments must be submitted on OLAT by 14:00 on their respective
due dates.

