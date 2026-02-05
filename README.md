  
# Chemical language models for generating compounds with triple-target activity    
## Decription
- Dual-target pre-training of a transformer model followed by triple-target fine-tuning to generate triple-target candidate compounds.
- The repository presents the code used in this publication[^1].
- Dual-target pre-training adapted from [^2].
- The CLM architecture was adapted from[^3], which modified the original code from [deep-molecular-optimization](https://github.com/MolecularAI/deep-molecular-optimization)

----------------------------------------------
## Usage
Create environment 

```
conda env create -f environment.yml
conda activate tt_trans
```
---------------------------------------------

**1. Preprocess data**

Build vocabulary and save the file. The output vocab.pkl is saved in the same folder as the input. The input file should contain SMILES strings of the source and target molecules under the columns 'Source_Mol' and 'Target_Mol', respectively.

```
python preprocess.py --input-data-folder <path_to_folder> --data-file-name <file_name.csv>

```

Example usage,
```
python preprocess.py --input-data-folder data --data-file-name ST_TT_data.csv
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
----------------------------------------------
**Main packages**
pandas-1.0.0
numpy-1.17.3
pytorch-1.4.0
rdkit-2020.03.2.0
scikit-learn-0.21.3
tensorboardx-2.0
----------------------------------------------

## References:
[^1]: Srinivasan, S., and Bajorath, J. (2026). Chemical language models for generating compounds with triple-target activity. Cell Reports Physical Science 7, 103054. https://doi.org/10.1016/j.xcrp.2025.103054. 
[^2]: Srinivasan, S., and Bajorath, J. (2025). Protocol to generate dual-target compounds using a transformer chemical language model. STAR Protocols 6, 103584. https://doi.org/10.1016/j.xpro.2024.103584.
[^3]:  Chen, H.; Vogt, M.; Bajorath, J. DeepAC – Conditional Transformer-Based Chemical Language Model for the Prediction of Activity Cliffs Formed by Bioactive Compounds. Digital Discovery 2022, 1, 898-909.

 
