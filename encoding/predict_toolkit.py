

import numpy as np
import torch  

def construct_word_embedding_table(word_dim,word_alphabet,unk_id,freeze,word_dict):

    scale = np.sqrt(3.0 / word_dim)
    table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
    table[unk_id, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
    oov = 0
    for word, index in word_alphabet.items():
        if  word_dict is not None:
            if word in word_dict :
                embedding = word_dict[word]
            elif word.lower() in word_dict and word_dict is not None:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
        else:
            embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    print('word OOV: %d' % oov)
    return torch.from_numpy(table)