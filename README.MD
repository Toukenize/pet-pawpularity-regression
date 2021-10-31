# Environment Setup

## 1. Dependencies
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2. Install environment with `conda install env.yml`

   - Alternative, if it doesn't work due to incompatible `cudatoolkit`, you can edit the `env.yml` by specifying a compatible `cudatoolkit` (more info [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)).

   - Utility function to export .yml : `conda env export | grep -v "^prefix: " > env.yml`
3. If no GPU is available on your environment, do the following to manually install the non-GPU version of dependencies:
    1. Create new environment `conda create -n paw-conda python=3.7 tensorflow pytorch -c pytorch`
    2. Install additional dependencies 
        > `pip install pandas ipykernel kaggle timm pytorch-lightning wandb scikit-learn opencv-python albumentations python-dotenv pydantic flake8 autopep8`


## 2. Jupyter Notebook

1. Activate your environment `conda activate paw-conda` 
2. Install the kernel `python -m ipykernel install --user --name "paw-conda" --display_name "paw-conda"`
3. Spin off your jupyter notebook as usual `jupyter notebook`

## 3. Config & Secrets

1. Refer to the in-line comments in `config.env` for details
2. To enable remote logging to WandB:
    1. Copy `secret_template.env`
    2. Rename it to `secret.env`
    3. Add your WandB API Key to the variable `WANDB_API_KEY`

# Run Training Script

1. With your environment activated and .env files set, run `python main_xxx.py`
