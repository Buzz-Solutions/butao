env_name=butao
python_version=3.10

conda create -n ${env_name} python=${python_version}
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $env_name

pip install -r requirements.txt
pip install -e .
pre-commit install
conda deactivate