import argparse
import os
from tqdm import tqdm
from config_preprocessing import *


# preprocessing used for the dalvik byte sequences. concatenates all lines and extracts 2-grams of them
def doc_preprocessing(s):
    N = 2
    lines = ''.join(s.splitlines())
    res = ' '.join([lines[i:i + N] for i in range(0, len(lines), N)])
    return res.split()


def get_dalvik_token_to_index(dalvik_opcode_path, save=False):
    tokens = [line.split()[1] for line in open(dalvik_opcode_path).readlines()]
    # index starts at 1 to have 0 for padding
    token_to_idx = dict(zip(tokens, range(1,len(tokens)+1)))
    if save:
        with open(os.path.join('token2idx_damd'), 'w') as f:
            for k,v in token_to_idx.items():
                print('{}:{}'.format(k,v), file=f)
    return token_to_idx


# converts files of dalvik opcode sequences into files of sequences of token indices where each token represents one
# dalvik opcode. Assumes that the file in document_path folder have ending .1 for malicious and .0 for benign.
# requires file containing all dalvid opcodes for conversion
def convert_docs_to_idx(document_path, dalvik_opcode_path, saving_path):
    token_to_idx = get_dalvik_token_to_index(dalvik_opcode_path, save=True)
    print('Converting {} files to tokenized representation...'.format(len(os.listdir(document_path))))
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    for fn in tqdm(os.listdir(document_path)):
        tokens = doc_preprocessing(open(os.path.join(document_path, fn), 'r').read())
        with open(os.path.join(saving_path, fn), 'w') as f:
            print(','.join([str(token_to_idx[idx]) for idx in tokens]), file=f)


if __name__ == '__main__':
    convert_docs_to_idx(opcode_path, dalvik_opcode_path, save_path)
