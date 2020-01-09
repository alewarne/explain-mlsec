import argparse
import contextlib
import itertools
import json
import os
from gensim.models.word2vec import Word2Vec
from config_word_to_vec import *


def train_word2vec(gadgets, vector_size=200, iter=100, workers=1):
    x = [gadget["tokens"] for gadget in gadgets]

    # Train Word2Vec
    w2v = Word2Vec(x, min_count=1, size=vector_size, iter=iter, workers=workers)
    return w2v


if __name__ == "__main__":
    with contextlib.ExitStack() as stack:
        f_list = [stack.enter_context(open(dataset)) for dataset in data_paths]
        gadgets = itertools.chain.from_iterable([json.load(f) for f in f_list])

    print("Training Word2Vec embedding...")
    w2v = train_word2vec(gadgets, embedding_dim, iterations, workers)

    print("Trained Word2Vec embedding with weights of shape:", w2v.wv.syn0.shape)
    if w2v_vocab_name:
        with open(w2v_vocab_name, 'w') as f:
            vocab = dict([(k, v.index) for k, v in w2v.wv.vocab.items()])
            f.write(json.dumps(vocab, indent=4, sort_keys=True))
    w2v.save(output_name)
    print("Written model to: {}".format(output_name))
