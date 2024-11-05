import pandas as pd
import os

from sklearn.model_selection import train_test_split

import utils.file as uf
import configuration.config_default as cfgd

SEED = 42
SPLIT_RATIO = 0.8


def get_smiles_list(file_name):
    """
    Get smiles list for building vocabulary
    :param file_name:
    :return:
    """
    pd_data = pd.read_csv(file_name, sep=",")

    print("Read %s file" % file_name)
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    print("Number of SMILES in chemical transformations: %d" % len(smiles_list))

    return smiles_list

def split_data(input_transformations_path, file_name, split_test = False, LOG=None):
    """
    Split data into training, validation or train, validation, test and write to files
    :param input_transformations_path:str
    :param split_test:bool
    """

    data = pd.read_csv(os.path.join(input_transformations_path, file_name), sep=",")
    if LOG:
        LOG.info("Read %s file" % input_transformations_path)

    if split_test:

        train, test = train_test_split(
            data, test_size=0.1, random_state=SEED)
        train, validation = train_test_split(train, test_size=0.1, random_state=SEED)

        if LOG:
            LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

        train.to_csv(os.path.join(input_transformations_path, "train.csv"), index=False)
        validation.to_csv(os.path.join(input_transformations_path, "validation.csv"), index=False)
        test.to_csv(os.path.join(input_transformations_path, "test.csv"), index=False)

    
    else:
        train, validation = train_test_split(data, test_size=0.1, random_state=SEED)
        if LOG:
            LOG.info("Train, Validation: %d, %d" % (len(train), len(validation)))

        
        train.to_csv(os.path.join(input_transformations_path, "train.csv"), index=False)
        validation.to_csv(os.path.join(input_transformations_path, "validation.csv"), index=False)
