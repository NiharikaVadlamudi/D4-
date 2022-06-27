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
## Generating CDDD Descriptors 
Given a .csv file with SMILES column , one can now generate the corresponding descriptor files , by running the following bash files & placing the datafiles in appropriate directory . 

```bash
cd cddd/cddd
bash train_reg_model.sh 
SEE THIS 
```

## Network Inference
Given a .csv file , with only SMILEs as the input , one has to follow thse steps : 



## Train the network 
Assuming , the CDDD descriptor file for both validation & train have been generated , the follow commands help us in 




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


