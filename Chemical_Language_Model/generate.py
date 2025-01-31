import numpy as np
import pickle as pkl
import os
import argparse
import pandas as pd
import torch

import utils.chem as uc
import utils.torch_util as ut
import utils.log as ul
import utils.plot as up
import configuration.config_default as cfgd
import models.dataset as md
import preprocess.vocabulary as mv
import configuration.opts as opts
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
from models.transformer.module.decode import decode
from models.transformer.encode_decode.model import EncoderDecoder

def split_batches(dataloader, num_gpus):
    """
    Split batches for multiple GPUs
    :param dataloader: dataloader
    :param num_gpus: number of GPUs
    :return: list of batches 
    """
    batches = list(dataloader)
    split_size = len(batches) // num_gpus
    return [batches[i * split_size: (i + 1) * split_size] for i in range(num_gpus)]

class GenerateRunner():

    def __init__(self, opt):

        self.save_path = os.path.join(opt.save_directory, opt.test_file_name,
                                      f'evaluation_{opt.epoch}')
        global LOG
        LOG = ul.get_logger(name="generate",
                            log_path=os.path.join(self.save_path, 'generate.log'))
        LOG.info(opt)
        LOG.info("Save directory: {}".format(self.save_path))

        # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        self.vocab = vocab
        self.tokenizer = mv.SMILESTokenizer()

    def initialize_dataloader(self, opt, vocab, test_file, device_ids):
        """
        Initialize dataloader
        :param opt:
        :param vocab: vocabulary
        :param test_file: test_file_name
        :return:
        """

        # Read test
        data = pd.read_csv(os.path.join(opt.data_path, test_file + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=self.tokenizer, prediction_mode=True)
        dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size,
                                                 shuffle=False, collate_fn=md.Dataset.collate_fn,
                                                 num_workers=len(device_ids) * 2,
                                                 pin_memory=True)
        return dataloader
    
    def process_batches(self, device_id, batches, model, opt, max_len, df_list, smiles_list, total_count, valid_count):
        device = f"cuda:{device_id}"
        model.to(device)
        model.eval()
        for i,batch in enumerate(ul.progress_bar(batches, total=len(batches))):
            src, source_length, _, src_mask, _, _, df = batch
            src = src.to(device)
            src_mask = src_mask.to(device)
            smiles, total, only_valid = self.sample(model, src, src_mask,
                                                                       source_length,
                                                                       opt.decode_type,
                                                                       num_samples=opt.num_samples,
                                                                       max_len=max_len,
                                                                       device=device,
                                                                       temperature=opt.temperature)
            df_list.append(df)
            smiles_list.extend(smiles)
            total_count.extend(total)
            valid_count.extend(only_valid)
        

    def generate(self, opt):

        # Get device IDs from the user
        device_ids = list(map(int, opt.cuda_device.split(',')))
        print(f"Requested device IDs: {device_ids}")

        # Set CUDA_VISIBLE_DEVICES if specified
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        # Adjust device IDs for DataParallel (relative indexing)
        device_ids = list(range(len(device_ids)))
        print(f"Adjusted device IDs for DataParallel: {device_ids}")

        # Data loader
        dataloader_test = self.initialize_dataloader(opt, self.vocab, opt.test_file_name, device_ids)
        total_count = []
        valid_count = []

        # Load model
        
        file_name = os.path.join(opt.model_path, f'model_{opt.epoch}.pt')
        
        model = EncoderDecoder.load_from_file(file_name)

        max_len = cfgd.DATA_DEFAULT['max_sequence_length']
        df_list = []
        sampled_smiles_list = []

        batches_ = split_batches(dataloader_test, len(device_ids))
        manager = Manager()
        shared_df_list = manager.list()
        shared_smiles_list = manager.list()
        shared_total_count = manager.list()
        shared_valid_count = manager.list()

        # Launch processes
        processes = []
        for i, device_id in enumerate(device_ids):
            p = Process(target=self.process_batches, args=(device_id, batches_[i], model, opt, max_len, 
                                                           shared_df_list, shared_smiles_list, shared_total_count, shared_valid_count))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

            df_list = list(shared_df_list)
            sampled_smiles_list = list(shared_smiles_list)
            total_count = list(shared_total_count)
            valid_count = list(shared_valid_count)
            
        # prepare dataframe
        data_sorted = pd.concat(df_list)
        sampled_smiles_list = np.array(sampled_smiles_list)

        for i in range(opt.num_samples):
            data_sorted['Predicted_smi_{}'.format(i + 1)] = sampled_smiles_list[:, i]
        
        data_sorted['Total_count'] = total_count
        data_sorted['Valid_count'] = valid_count

        # Save generated molecules
        result_path = os.path.join(self.save_path, "generated_molecules.csv")
        LOG.info("Save to {}".format(result_path))
        data_sorted.to_csv(result_path, index=False)

    def sample(self, model, src, src_mask, source_length, decode_type, num_samples=10, # num_samples=50 from opts - number of molecules to be generated
               max_len=cfgd.DATA_DEFAULT['max_sequence_length'],
               device=None, temperature=1.0):
        batch_size = src.shape[0] # batch size is the number of source molecules in the batch
        num_only_valid = np.zeros(batch_size)  
        num_valid_batch = np.zeros(batch_size)  # current number of unique and valid samples out of total sampled
        num_valid_batch_total = np.zeros(batch_size)  # current number of sampling times no matter unique or valid
        num_valid_batch_desired = np.asarray([num_samples] * batch_size)
        unique_set_num_samples = [set() for i in range(batch_size)]   # set of unique mols for each starting molecule
        batch_index = torch.LongTensor(range(batch_size)) # [0,1,2,3] if batch_size is 4
        batch_index_current = torch.LongTensor(range(batch_size)).to(device)
        ## to keep track of which indices that need further sampling
        start_mols = []
        # zeros correspondes to ****** which is valid according to RDKit
        sequences_all = torch.ones((num_samples, batch_size, max_len))
        sequences_all = sequences_all.type(torch.LongTensor)
        max_trials = 100  # Maximum trials for sampling
        current_trials = 0

        if decode_type == 'greedy':
            max_trials = 1

        # Set of unique starting molecules
        if src is not None:
            start_ind = 0
            for ibatch in range(batch_size): # ibatch is the index of the source molecule in the batch
                source_smi = self.tokenizer.untokenize(self.vocab.decode(src[ibatch].tolist()[start_ind:])) # one source molecule from src
                source_smi = uc.get_canonical_smile(source_smi)
                unique_set_num_samples[ibatch].add(source_smi)
                start_mols.append(source_smi)

        with torch.no_grad():
            while not all(num_valid_batch >= num_valid_batch_desired) and current_trials < max_trials:
                current_trials += 1

                # batch input for current trial
                if src is not None:
                    src_current = src.index_select(0, batch_index_current) 
                if src_mask is not None:
                    mask_current = src_mask.index_select(0, batch_index_current)
                batch_size = src_current.shape[0]

                # sample molecule
                sequences = decode(model, src_current, mask_current, max_len, decode_type, temperature=temperature) # calling the decoder 

                padding = (0, max_len-sequences.shape[1],
                            0, 0)
                sequences = torch.nn.functional.pad(sequences, padding)

                # Check valid and unique
                smiles = []  ## stores all the smiles generated
                is_valid_index = []
                batch_index_map = dict(zip(list(range(batch_size)), batch_index_current))
                # Valid, ibatch index is different from original, need map back
                for ibatch in range(batch_size):
                    seq = sequences[ibatch]
                    smi = self.tokenizer.untokenize(self.vocab.decode(seq.cpu().numpy()))
                    smi = uc.get_canonical_smile(smi)
                    smiles.append(smi)
                    # valid and not same as starting molecules
                    if uc.is_valid(smi):
                        is_valid_index.append(ibatch)
                    # total sampled times
                    num_valid_batch_total[batch_index_map[ibatch]] += 1

                # Check if duplicated and update num_valid_batch and unique
                for good_index in is_valid_index:
                    index_in_original_batch = batch_index_map[good_index]
                    num_only_valid[index_in_original_batch] += 1
                    if smiles[good_index] not in unique_set_num_samples[index_in_original_batch]:
                        unique_set_num_samples[index_in_original_batch].add(smiles[good_index])
                        num_valid_batch[index_in_original_batch] += 1

                        sequences_all[int(num_valid_batch[index_in_original_batch] - 1), index_in_original_batch, :] = \
                            sequences[good_index]

                not_completed_index = np.where(num_valid_batch < num_valid_batch_desired)[0]
                if len(not_completed_index) > 0:
                    batch_index_current = batch_index.index_select(0, torch.LongTensor(not_completed_index)).to(device)

        # Convert to SMILES
        smiles_list = [] # [batch, topk]
        seqs = np.asarray(sequences_all.numpy())
        # [num_sample, batch_size, max_len]
        batch_size = len(seqs[0])
        for ibatch in range(batch_size):
            topk_list = []
            for k in range(num_samples):
                seq = seqs[k, ibatch, :]
                topk_list.extend([self.tokenizer.untokenize(self.vocab.decode(seq))])
            smiles_list.append(topk_list)


        return smiles_list, num_valid_batch_total, num_only_valid
     

def run_main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='generate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.generate_opts(parser)
    opt = parser.parse_args()
    runner = GenerateRunner(opt)
    runner.generate(opt)


if __name__ == "__main__":
    # Set up multiprocessing start method to spawn
    mp.set_start_method('spawn', force=True) # for subprocessing to star t CUDA in clean state
    mp.set_sharing_strategy('file_system')
    run_main()
