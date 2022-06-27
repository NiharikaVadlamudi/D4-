#!/bin/bash
#SBATCH --job-name="CDDD:Descriptor Generation"
#SBATCH -A $USER
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=generate_reg_cddd_descriptors_task.txt
#SBATCH --mail-type=END

python3 run_cddd.py --use_gpu --input reg_admet_train.csv --output reg_train_descriptors.csv  --smiles_header molecule_smiles
python3 run_cddd.py --use_gpu --input reg_admet_test.csv --output reg_test_descriptors.csv  --smiles_header molecule_smiles
