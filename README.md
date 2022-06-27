# D4-MP 

Description : This repository will consist baseline experiments for Multi-Task Molecular Property Prediction on Harvard's TDC dataset . 

## Installation
Easy installation via [conda](https://www.anaconda.com/) : 
```bash
conda env create --file cddd.yml 
conda activate cddd
```

## ADMET Task Data Generation 
A total of 6 regression tasks - Caco-2 , Lipophilicity , Solubility (AqSolDB) , PPBR , Acute Toxicity LD50 & Clearance (Hepatocyt) are catergorised under regression. To generate the datafiles ( train & test ) , run the following commands : 
```bash
cd tdc_regression
bash reg_data_generation.sh 
```
For training the network : 

```bash
cd tdc_regression
bash train_reg_model.sh 
```

## Demo File 
For showcasing a demo of the model , please use the .csv files with CDDD descriptors to save time . 
```bash
python3 testing.py --csv_file 'test_results.csv' --checkpointfile 'optimal_checkpoint.pth'
```

## Generating CDDD Descriptors 
Given a .csv file with SMILES column , one can now generate the corresponding descriptor files , by running the following bash files & placing the datafiles in appropriate directory . 
Make sure you place your csv files in data_files_admet 
```bash
cd cddd/cddd
bash descriptor_batch.sh 
```

## Network Inference
Given a .csv file , with only SMILEs as the input , one has to follow thse steps : 
```bash
cd cddd/cddd
python3 run_cddd.py --use_gpu --input <csvFileName> --output <outputDescriptorFile>  --smiles_header molecule_smiles
```
* Move the outputDescriptorFile outside cddd/ folder , and test it with : 
```bash
python3 testing.py --csv_file <outputFileName> --mode 'test' --checkpointfile 'optimal_checkpoint'

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


