import gzip
import codecs
from tqdm import tqdm
import numpy as np
np.random.seed(123)
import torch

from encoding.constants import UNK_ID, DIGIT_RE

def construct_word_embedding_table(word_dim, word_dictionary, word_embed):
  scale = np.sqrt(3.0 / word_dim)
  table = np.empty([word_dictionary.size(), word_dim], dtype=np.float32)
  table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
  oov = 0
  for word, index in word_dictionary.items():
    if word in word_embed:
      embedding = word_embed[word]
    elif word.lower() in word_embed:
      embedding = word_embed[word.lower()]
    else:
      embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
      oov += 1
    table[index, :] = embedding
  print('word OOV: %d/%d' % (oov, word_dictionary.size()))
  return torch.from_numpy(table)

def load_word_embeddings(path, dry_run):
  embed_dim = -1
  embed_dict = dict()
  pbar = None
  with codecs.open(path, 'r', 'utf-8') as file:
    li = 0
    for line in file:
      line = line.strip()
      if len(line) == 0:
        continue
      tokens = line.split()
      if len(tokens) < 3:
        pbar=tqdm(total=int(tokens[0]) if not dry_run else 100)
        continue
      if embed_dim < 0:
        embed_dim = len(tokens) - 1
      else:
        assert (embed_dim + 1 == len(tokens))
      embed = np.empty([1, embed_dim], dtype=np.float32)
      embed[:] = tokens[1:]
      word = DIGIT_RE.sub(b"0", str.encode(tokens[0])).decode()
      embed_dict[word] = embed
      li = li + 1
      if li%50==0:
        pbar.update(50)
      if dry_run and li==100:
        break
  pbar.close()
  return embed_dict, embed_dim

class Sentence(object):
  def __init__(self, words, word_ids, char_seqs, char_id_seqs, lines,
               word_lenght=None, feats = None,lexicon_tags=None):
    self.words = words
    self.word_ids = word_ids
    self.char_seqs = char_seqs
    self.char_id_seqs = char_id_seqs
    self.lexicon_tags = lexicon_tags
    self.lines = lines
    self.feats = feats
    self.word_lengh=word_lenght

  def length(self):
    return len(self.words)

class DependencyInstance(object):
  def __init__(self, sentence, postags, pos_ids, heads, types, type_ids):
    self.sentence = sentence
    self.postags = postags
    self.pos_ids = pos_ids
    self.heads = heads
    self.types = types
    self.type_ids = type_ids

  def length(self):
    return self.sentence.length()


  