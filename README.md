  
# Design of Dual-Target Compounds Using a Transformer Chemical Language Model    
## Decription
- A transformer-based chemical language model(CLM) was implemented to generate dual-target compounds(DT-CPDs) from single-target compounds (ST-CPDs).
- The repository presents the code used in this publication[^1].
- The CLM architecture was adapted from[^2], which modified the original code from [deep-molecular-optimization](https://github.com/MolecularAI/deep-molecular-optimization)

----------------------------------------------
## Usage
Create environment 

```
conda env create -f environment.yml
conda activate dt_trans
```
---------------------------------------------

**1. Preprocess data**

Build vocabulary and save the file. The output vocab.pkl is saved in the same folder as the input. The input file should contain SMILES strings of the source and target molecules under the columns 'Source_Mol' and 'Target_Mol', respectively.

```
python preprocess.py --input-data-folder <path_to_folder> --data-file-name <file_name.csv>

```

Example usage,
```
python preprocess.py --input-data-folder data --data-file-name ST_DT_data.csv
```


**2. Split train data**
Split train data into train and validation for training.

```
python split_data.py --input-data-folder <path_to_folder> --data-file-name <file_name.csv>
```

**3. Train model**

 Train the model and save results. Make sure that the vocabulary is in the same folder as the input files.
```
python train.py --model-choice transformer --data-path <path_to_folder> --save-directory <path_to_folder>
``` 

**4. Generate molecules**

Use the model saved at a given epoch to generate molecules for the given test file, and save the results. 

```
python generate.py --model-choice transformer --data-path <path_to_folder> --test-file-name <test_file_name> --model-path <path_to_model> --epoch <epoch_number>
```   

## References:
[^1]: Srinivasan, S.; Bajorath, J. Generation of Dual-Target Compounds Using a Transformer Chemical Language Model. Cell Reports Physical Science 2024, 102255. https://doi.org/10.1016/j.xcrp.2024.102255.  
[^2]:  Chen, H.; Vogt, M.; Bajorath, J. DeepAC – Conditional Transformer-Based Chemical Language Model for the Prediction of Activity Cliffs Formed by Bioactive Compounds. Digital Discovery 2022, 1, 898-909.

 
