"""
Preprocess
- build vocabulary
- split data into train, validation
"""
import os
import argparse
import pickle

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf

global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: Build vocabulary")

    parser.add_argument("--input-data-folder", "-i", help=("Path to the input file folder"), type=str, required=True)
    parser.add_argument("--data-file-name", "-d", help=("File name of the data file"), type=str, required=True)

    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()


    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    smiles_list = pdp.get_smiles_list(os.path.join(args.input_data_folder, args.data_file_name))
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer)
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

    # Save vocabulary to file
    
    output_file = os.path.join(args.input_data_folder, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))



