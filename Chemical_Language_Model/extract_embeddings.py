import os
import argparse
import pickle
import torch
import torch.nn
from models.transformer.encode_decode.model import EncoderDecoder
from preprocess.vocabulary import Vocabulary

def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Extraxt the embeddings from the model")

    parser.add_argument("--model-path", "-m", help=("Path to the model"), type=str, required=True)
    parser.add_argument("--vocab-path", "-v", help=("Path to the vocabulary"), type=str, required=True)
    parser.add_argument("--save-path", "-s", help=("Path to save the embeddings"), type=str, required=True)
    parser.add_argument("--epoch", "-e", help=("Epoch of the model"), type=int, required=True)
    parser.add_argument("--type", "-t", help=("Type of embeddings to extract"), type=str, required=False, default='target')

    return parser.parse_args()
    

def get_embeddings(model, type='target'):
    if type == 'source':
        src_embeddings = model.src_embed[0].lut.weight.detach().cpu().numpy()
        return src_embeddings
    elif type == 'target':
        target_embeddings = model.tgt_embed[0].lut.weight.detach().cpu().numpy()
        return target_embeddings
    

def save_embeddings(embedding, save_path, type='target'):
    if type == 'source':
        save_path = save_path + 'source.pkl'
    elif type == 'target':
        save_path = save_path + 'target.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(embedding, f)

def get_tokens(file_path):
    with open(os.path.join(file_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return vocab.tokens()

def save_tokens(tokens, save_path):
    with open(save_path + 'tokens.pkl', 'wb') as f:
        pickle.dump(tokens, f)


if __name__ == '__main__':
    
    args = parse_args()

    file_name = os.path.join(args.model_path, f'model_{args.epoch}.pt')
    model = EncoderDecoder.load_from_file(file_name)
    emb = get_embeddings(model, type=args.type)
    save_embeddings(emb, args.save_path, type=args.type)
    tok = get_tokens(args.vocab_path)
    save_tokens(tok, args.save_path)