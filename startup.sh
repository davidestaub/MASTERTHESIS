module load stack/2024-06
module load gcc/12.2.0
module load python/3.11.6
module load python_cuda/3.11.6
module load cuda/12.1.1
module load eth_proxy
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --user
pip install --user devito
wandb login
