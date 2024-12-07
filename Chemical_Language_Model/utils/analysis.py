import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class TransformerAnalysis:
    
    def __init__(self, df):
        self.df = df
        self.DT_test_set = set(df['Target_Mol'])
        self.DT_test_set = set([Chem.CanonSmiles(smi) for smi in self.DT_test_set]) # canonicalization
        self.predictions = self._extract_all_predictions()

    def _extract_all_predictions(self):
        """Extract predictions for all rows in df and store as a set."""
        all_pred = set()
        for i in range(len(self.df)):
            row_preds = self._extract_predictions(self.df.iloc[i])
            all_pred.update(row_preds)

        all_pred = [Chem.CanonSmiles(smi) for smi in all_pred]
        return all_pred

    def _extract_predictions(self, row):
        """Extracts predictions from a row, filtering for string values."""
        pred_list = row[2:]  # Skip first two columns
        return [pred for pred in pred_list if isinstance(pred, str)]

    def _calculate_percentage(self, numerator, denominator):
        """Calculates percentage with safe division, returns 0 if denominator is 0."""
        return round((numerator / denominator) * 100, 2) if denominator else 0

    def overall_reproducibility(self):
        """
        Calculate the overall reproducibility for the test set.
        return: len(self.DT_test_set): Number of unique molecules in the test set.
        return: count_reprod: Number of unique molecules in the test set that were reproduced.
        return: repro: Percentage of unique molecules in the test set that were reproduced.
        """
        count_reprod = len(self.DT_test_set.intersection(self.predictions))
        repro = self._calculate_percentage(count_reprod, len(self.DT_test_set))
        return len(self.DT_test_set), count_reprod, repro

    def average_validity_uniqueness(self):
        """
        Calculate validity and uniqueness percentages, averaged over the test set.
        """
        # Calculate validity and uniqueness percentage for each row
        self.df['percentage_validity'] = self.df.apply(
            lambda x: self._calculate_percentage(x['Valid_count'], x['Total_count']), axis=1
        )

        # Calculate uniqueness based on predictions
        self.df['num_uniq'] = self.df.apply(
            lambda x: len(set(self._extract_predictions(x))), axis=1
        )
        self.df['percentage_uniqueness'] = self.df.apply(
            lambda x: self._calculate_percentage(x['num_uniq'], x['Valid_count']), axis=1
        )

        return round(self.df['percentage_validity'].mean(), 2), round(self.df['percentage_uniqueness'].mean(), 2)

    def novel_predictions(self, df_train):
        """
        Calculate the average number of novel molecules predicted per source molecule.
        Requires `df_train` dataframe with known training molecules.
        param: df_train- training set
        return: novel_num: Number of novel molecules predicted per source molecule.
        return: novel_percent: Percentage of novel molecules predicted per source molecule. 
        """

        DT_train_set = set(df_train['Target_Mol'])
        DT_train_set = set([Chem.CanonSmiles(smi) for smi in DT_train_set])
        novel_num = []
        novel_percent = []

        for i in range(len(self.df)):
            pred_set = set(self._extract_predictions(self.df.iloc[i]))
            novel = pred_set.difference(DT_train_set)
            novel_num.append(len(novel))
            novel_percent.append(self._calculate_percentage(len(novel), len(pred_set)))

        self.df['num_novel'] = novel_num
        self.df['novel_percent'] = novel_percent

        return round(self.df['novel_percent'].mean(), 2)
    

    def calculate_tanimoto_similarity(self,df_train):
        """
        Calculate the average Tanimoto similarity and the nearest neighbor similarity
        for a given compound with all training compounds.
        
        param: df_train - Training set dataframe with a 'Target_Mol' column containing SMILES strings.
        return: avg_similarity - Average Tanimoto similarity with all training compounds.
        return: nearest_neighbor_similarity - Tanimoto similarity to the nearest neighbor in the training set.
        """
        # Generate fingerprints
        m = [Chem.MolFromSmiles(smi) for smi in self.DT_test_set]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in m if m]
        
        # Generate fingerprints for all training compounds
        DT_train_set = list(df_train['Target_Mol'].unique())
        train_mols = [Chem.MolFromSmiles(smi) for smi in DT_train_set]
        train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols if m]
        
        # Compute Tanimoto similarities
        average_similarity = []
        nearest_neighbor_similarity = []
        nn_smiles = []
        for fp in fps:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            avg_sim = round(np.mean(similarities), 4)
            nn_sim  = round(max(similarities), 4)
            nn_index = np.argmax(similarities)
            nn_smiles.append(DT_train_set[nn_index])
            average_similarity.append(avg_sim)
            nearest_neighbor_similarity.append(nn_sim)
        
        sim_df = pd.DataFrame({'Generated_Molecule': list(self.DT_test_set),
                               'Average_Similarity': average_similarity,
                               'Nearest_Neighbor_Similarity': nearest_neighbor_similarity,
                               'Nearest_Neighbor': nn_smiles})

        return sim_df