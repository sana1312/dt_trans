import os
import argparse

import preprocess.data_preparation as dp
import utils.log as ul

global LOG
LOG = ul.get_logger("split_data", "experiments/split_data.log")


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Split data: Split data into train,validation test or train, validation")

    parser.add_argument("--input-data-folder", "-i", help=("Path to the input file folder"), type=str, required=True)
    parser.add_argument("--data-file-name", "-d", help=("File name of the data file"), type=str, required=True)
    parser.add_argument("--split-test", "-s", help=("True for split test set"), type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

# Split data into train, validation
dp.split_data(args.input_data_folder, args.data_file_name, args.split_test, LOG)