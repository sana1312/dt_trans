import os
import argparse
import pandas as pd
import numpy as np

from utils.analysis import TransformerAnalysis

def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Transformer Analysis: Analyze the results of the Transformer model")

    parser.add_argument("--result-path", "-r", help=("Path to the results file"), type=str, required=True)
    parser.add_argument("--train-path", "-t", help=("Path to the training file"), type=str, required=True)
    parser.add_argument("--save-directory", "-s", help=("Path to the save directory"), type=str, default='analysis')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Load results
    results = pd.read_csv(args.result_path)
    train = pd.read_csv(args.train_path)

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    analysis = TransformerAnalysis(results)
    unique_targ, count_reproduced, percentage_reproduced = analysis.overall_reproducibility()
    validity, uniqueness = analysis.average_validity_uniqueness()
    novel_predictions = analysis.novel_predictions(train)
    simillarity_df = analysis.calculate_tanimoto_similarity(train)

    # write the statistics to a file
    with open(os.path.join(args.save_directory, 'analysis.txt'), 'w') as f:
        f.write(f"Overall Reproducibility: {count_reproduced}/{unique_targ} ({percentage_reproduced}%)\n")
        f.write(f"Average Validity: {validity}%\n")
        f.write(f"Average Uniqueness: {uniqueness}%\n")
        f.write(f"Average Novelty: {novel_predictions}")

    simillarity_df.to_csv(os.path.join(args.save_directory, 'similarity.tsv'), sep='\t', index=False)

    print(f"Overall Reproducibility: {count_reproduced}/{unique_targ} ({percentage_reproduced}%)")
    print(f"Average Validity: {validity}%")
    print(f"Average Uniqueness: {uniqueness}%")
    print(f"Average Novelty: {novel_predictions}")
