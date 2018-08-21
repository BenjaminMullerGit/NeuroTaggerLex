# -*- coding: utf-8 -*-

import sys
import codecs
import json
from encoding.constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS, ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE
from i_o.conllu_reader import CoNLLReader, preprocess_feats
from encoding.dictionary import Dictionary

import numpy as np
np.random.seed(123)
import torch
torch.manual_seed(123)
from torch.autograd import Variable
import re
import os
DIGIT_RE = re.compile(br"\d")

def expand_vocab(data_paths,
                 vocab_list,
                 word_embed_dict,
                 word_dictionary,
                 char_dictionary,
                 pos_dictionary,
                 type_dictionary,
                 lexicon_dictionary,
                 lexicon,
                 singletons,
                 expand_word_dic=False,
                 expand_char_dic=False,
                 expand_pos_dic=False,
                 expand_lexicon_dic=False,
                 expand_type_dic=False,
                 env="python3", dry_run =False):
    """
    Designed to expand word_embedding in every cases --> Not pos (not to mess up everything)
    :param data_paths:
    :param vocab_list:
    :param word_embed_dict:
    :param lexicon:
    :param char_dictionary:
    :param pos_dictionary:
    :param type_dictionary:
    :param lexicon_dictionary:
    :param env:
    :param dry_run:
    :return:
    """
    if lexicon is not None :
      assert pos_dictionary is not None
    vocab_set = set(vocab_list)
    for data_path in data_paths:
      with codecs.open(data_path, 'r', 'utf-8') as file:
        li = 0
        for line in file:
          line = line.strip()
          if len(line) == 0 or line[0]=='#':
            continue
          tokens = line.split('\t')
          if '-' in tokens[0] or '.' in tokens[0]:
            continue
          if expand_char_dic:
            for char in tokens[1]:
              char_dictionary.add(char)
          if env == "python3":
            word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
          else:
            word = DIGIT_RE.sub(b"0", tokens[1].encode('utf-8', 'replace'))
          pos = tokens[3] if tokens[4] == '_' else tokens[3]+'$$$'+tokens[4]
          typ = tokens[7]
          if expand_pos_dic:
            pos_dictionary.add(pos)
          if expand_type_dic:
            type_dictionary.add(typ)
          # filling lexicon dictionary with words  unseen so far
          if expand_lexicon_dic:
            assert lexicon is not None, "ERROR : lexicon required for expanding lexicon dic"
            if lexicon_dictionary.get(word,True):
                if lexicon.get(word) is not None:
                  pos_labels = lexicon.get(word)
                  lexicons_pos_ids = [pos_dictionary.get_index(pos_lexicon) for pos_lexicon in pos_labels]
                  lexicon_dictionary[word] = lexicons_pos_ids
                elif lexicon.get(word.lower()) is not None:
                  pos_labels = lexicon.get(word.lower())
                  lexicons_pos_ids = [pos_dictionary.get_index(pos_lexicon) for pos_lexicon in pos_labels]
                  # WORD ? really like this o word.lower()
                  lexicon_dictionary[word] = lexicons_pos_ids

          if expand_word_dic and word_embed_dict is not None :
            #assert word_embed_dict is not None, "ERROR word_embed_dic required to expand word vectors "
            # adding to vocabulary words that appear in the expansion data that are in the lexicon
            if word not in vocab_set and (word in word_embed_dict or word.lower() in word_embed_dict):
              vocab_set.add(word)
              vocab_list.append(word)
          li = li + 1
          if dry_run and li==100:
            break
          if expand_lexicon_dic:
            assert lexicon is not None, "ERROR word_embed_dic required to expand word vectors "
            # adding to vocabulary words that appear in the expansion data that are in the lexicon
            if word not in vocab_set and (word in lexicon or word.lower() in lexicon):
              vocab_set.add(word)
              vocab_list.append(word)

          # fill word dictionary with expanded vocabulary list
          if expand_word_dic:
            for word in vocab_list:
                  word_dictionary.add(word)
                  if word in singletons:
                    word_dictionary.add_singleton(word_dictionary.get_index(word))

    return word_dictionary, char_dictionary, pos_dictionary, type_dictionary, lexicon_dictionary


      #self.morph_full_feats.add(feat)
      #self.morph_class_dictionary.add(morph_class)

def create_dict(dict_path, train_path, dev_path, test_path,
                word_embed_dict, dry_run,features=False, lexicon_feats=None,
                lexicon_feats_inst2id=None,lexicon=None, env="python2",
                expand=False):

  assert env in ["python2","python3"]

  word_dictionary = Dictionary('word', default_value=True, singleton=True)
  char_dictionary = Dictionary('character', default_value=True)
  pos_dictionary = Dictionary('pos', default_value=True, keep_growing=False)
  type_dictionary = Dictionary('type', default_value=True)
  if lexicon is not None:
    lexicon_dictionary = {} #Dictionary('lexicon', default_value=True)
  else:
    lexicon_dictionary = None
  if features:
    feature_dictionary = Dictionary('feats', default_value=True, keep_growing=True)
  else:
    feature_dictionary = None

  # default by hand
  ##lexicon_dictionary['<_UNK>'] = [0]
  ##lexicon_dictionary[ROOT_POS] = [1]
  #lexicon_dictionary.add()
  char_dictionary.add(PAD_CHAR)
  pos_dictionary.add(PAD_POS)
  type_dictionary.add(PAD_TYPE)

  if features:
    feature_dictionary.add(PAD_POS)
  if lexicon_feats:
    lexicon_feats_dictionary = {}
  else:
    lexicon_feats_dictionary = None

  char_dictionary.add(ROOT_CHAR)
  pos_dictionary.add(ROOT_POS)
  type_dictionary.add(ROOT_TYPE)
  if features:
    feature_dictionary.add(ROOT_POS)

  char_dictionary.add(END_CHAR)
  pos_dictionary.add(END_POS)
  type_dictionary.add(END_TYPE)
  if features:
    feature_dictionary.add(END_POS)

  vocab = dict()

  if not os.path.isdir(dict_path):
    os.mkdir(dict_path)

    print("INFO DICTIONARIES : CREATE ALPHABETS AGAIN")

    with codecs.open(train_path, 'r', 'utf-8') as file:
      li = 0
      for line in file:
        line = line.strip()
        if len(line) == 0 or line[0]=='#':
          continue

        tokens = line.split('\t')
        if '-' in tokens[0] or '.' in tokens[0]:
          continue

        for char in tokens[1]:
          char_dictionary.add(char)
        #GANESH orginal
        if env == "python3":
          word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
        else:
          word = DIGIT_RE.sub(b"0", tokens[1].encode('utf-8','replace'))
        pos = tokens[3] if tokens[4] == '_' else tokens[3]+'$$$'+tokens[4]
        typ = tokens[7]
        feats = tokens[5]
        # reorder and filter feats
        if features:
          feats = preprocess_feats(feats)
          feature_dictionary.add(feats)

        pos_dictionary.add(pos)
        type_dictionary.add(typ)

        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1

        li = li + 1
        if dry_run and li == 100:
          break
      if features:
        feature_dictionary.keep_growing = False
    # collect singletons
    min_occurence = 1
    singletons = set([word for word, count in vocab.items() if count <= min_occurence])

    # if a singleton is in pretrained embedding dict, set the count to min_occur + c
    if word_embed_dict is not None:
      for word in vocab.keys():
        if word in word_embed_dict or word.lower() in word_embed_dict:
          vocab[word] += 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]

    max_vocabulary_size = 50000
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]

    # filling lexicon_dictionary with word of the training vocabulary
    expand_lexicon_dic = False
    if lexicon_feats:
      print("vocab list", len(vocab_list))
      for word in vocab_list:
        word_lower = word.lower()
        if lexicon_feats_inst2id.get(word) is not None:
          lexicon_feats_dictionary[word] = lexicon_feats_inst2id.get(word)
        elif lexicon_feats_inst2id.get(word_lower) is not None:
          lexicon_feats_dictionary[word] = lexicon_feats_inst2id.get(word_lower)

    if lexicon is not None:
      expand_lexicon_dic = True
      for word in vocab_list:
        #print(word )
        # !!--> LEXICON
        if isinstance(word, bytes):
          word_d = word.decode()
        else:
          word_d = word
        word_lower = word_d.lower()
        if lexicon.get(word_d) is not None:
          pos_labels = lexicon.get(word_d)
        elif lexicon.get(word_lower) is not None:
          pos_labels = lexicon.get(word_lower)
        else:
          continue
        if lexicon_dictionary.get(word_d, True):
          lexicons_pos_ids = [pos_dictionary.get_index(pos_lexicon) for pos_lexicon in pos_labels]
          lexicon_dictionary[word_d] = lexicons_pos_ids

    # EXPAND former defintion was here
    #expand_vocab([dev_path, test_path])
    #print("INFO lexicon_dic lenght {}".format(len(lexicon_dictionary)))
    word_dictionary, char_dictionary, \
     pos_dictionary, type_dictionary, \
     lexicon_dictionary = expand_vocab([dev_path],
                                       vocab_list=vocab_list,
                                       word_dictionary=word_dictionary,
                                       lexicon=lexicon,
                                       char_dictionary=char_dictionary,
                                       expand_lexicon_dic=expand_lexicon_dic,
                                       expand_word_dic=True,
                                       pos_dictionary=pos_dictionary,
                                       singletons=singletons,
                                       type_dictionary=type_dictionary,
                                       lexicon_dictionary=lexicon_dictionary,
                                       word_embed_dict=word_embed_dict)
    len_lexicon = len(lexicon_dictionary) if lexicon is not None else "NO"
    print("INFO lexicon_dic lenght {} after expansion".format(len_lexicon))
    #if False:
    ##  for word in vocab_list:
    #    word_dictionary.add(word)
    #    if word in singletons:
    #      word_dictionary.add_singleton(word_dictionary.get_index(word))
    if lexicon is not None:
      with open(dict_path+"/lexicon.json","w") as f:
        json.dump(lexicon_dictionary, f)
    if features:
      feature_dictionary.save(dict_path)
      print("SAVING feature_dictionary")
    if lexicon_feats:
      with open(dict_path+"/lexicon_feats.json","w") as f:
        json.dump(lexicon_feats_dictionary, f)
    word_dictionary.save(dict_path)
    char_dictionary.save(dict_path)
    pos_dictionary.save(dict_path)
    type_dictionary.save(dict_path)
  else:
    print("INFO DICTIONARIES : LOAD EXISTING DICTIONARIES")
    if lexicon:
       expand_lexicon_dic = True
       with open(dict_path+"/lexicon.json","r") as f:
        lexicon_dictionary = json.load(f)
    if lexicon_feats:
       with open(dict_path+"/lexicon_feats.json","r") as f:
        lexicon_feats_dictionary = json.load(f)
    else:
      expand_lexicon_dic = True 
    word_dictionary.load(dict_path, "word", keep_growing=True)
    char_dictionary.load(dict_path, "character")
    pos_dictionary.load(dict_path, "pos")
    type_dictionary.load(dict_path, "type")
    if features:
      feature_dictionary.load(dict_path,'feats')
      feature_dictionary.keep_growing = False
    if expand:
      if expand_lexicon_dic:
        print("INFO expanding Lexicon ")
        print("INFO lexicon_dic lenght {} before expansing ".format(len(lexicon_dictionary)))

      word_dictionary, char_dictionary, pos_dictionary, type_dictionary, \
        lexicon_dictionary = expand_vocab(data_paths=[test_path],
                                          vocab_list=list(word_dictionary.instance2index.keys()),
                                          char_dictionary=char_dictionary,
                                          word_dictionary=word_dictionary,
                                          expand_lexicon_dic=expand_lexicon_dic,
                                          lexicon=lexicon,
                                          singletons=word_dictionary.singletons,
                                          pos_dictionary=pos_dictionary,
                                          type_dictionary=type_dictionary,
                                          lexicon_dictionary=lexicon_dictionary,
                                          word_embed_dict=word_embed_dict)
      if expand_lexicon_dic:
        print("INFO lexicon_dic lenght {} after expansion ".format(len(lexicon_dictionary)))

  word_dictionary.close()
  char_dictionary.close()
  pos_dictionary.close()
  type_dictionary.close()

  if not lexicon and not features and not lexicon_feats:
    return word_dictionary, char_dictionary, pos_dictionary, type_dictionary
  elif lexicon and not features and not lexicon_feats:
    return word_dictionary, char_dictionary, (pos_dictionary, lexicon_dictionary), type_dictionary
  elif features and not lexicon_feats and not lexicon_feats:
    return word_dictionary, char_dictionary, (pos_dictionary, lexicon_dictionary, feature_dictionary), type_dictionary
  elif lexicon_feats:
    return word_dictionary, char_dictionary, (pos_dictionary, lexicon_dictionary, feature_dictionary, lexicon_feats_dictionary), type_dictionary




def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, max_size=None,
              feats_dictionary=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False, dry_run=False,
              lexicon_dictionary=None, word_lenghts_return=False,
              env="python2"):

  _buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, -1]
  last_bucket_id = len(_buckets) - 1
  data = [[] for _ in _buckets]
  max_char_length = [0 for _ in _buckets]
  print('Reading data from %s' % source_path)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary,feature_dictionary=feats_dictionary,
                       lexicon_dictionary=lexicon_dictionary)
  inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, env=env, word_len=word_lenghts_return)
  ids = []
  while inst is not None and (not dry_run or counter < 100):
    inst_size = inst.length()
    sent = inst.sentence
    for bucket_id, bucket_size in enumerate(_buckets):
      if inst_size < bucket_size or bucket_id == last_bucket_id:
        ids.append(counter)
        # lexicon only
        _data_complete = [sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, counter, sent.words, sent.lines, sent.lexicon_tags, sent.feats, sent.word_lengh]
        char_lenghts = [len(char_seq) for char_seq in sent.char_seqs]
        max_len = max(char_lenghts)
        data[bucket_id].append(_data_complete)
        if max_char_length[bucket_id] < max_len:
          max_char_length[bucket_id] = max_len
        if bucket_id == last_bucket_id and _buckets[last_bucket_id]<len(sent.word_ids):
          _buckets[last_bucket_id] = len(sent.word_ids)
        break
    inst = reader.getNext(normalize_digits=normalize_digits,
                          symbolic_root=symbolic_root,
                          symbolic_end=symbolic_end,word_len=word_lenghts_return,
                          env=env)
    counter += 1
  reader.close()
  return data, max_char_length, _buckets




def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary,
                          feats_lex_dic=None, feats_dictionary=None,max_size=None, normalize_digits=True,
                          symbolic_root=False,
                          symbolic_end=False, use_gpu=False, volatile=False, dry_run=False,debug=False,
                          lexicon_dictionary=None, word_lenghts_return=False,
                          env="python2"):

  data, max_char_length, _buckets = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary,
                                              max_size=max_size, normalize_digits=normalize_digits,
                                              lexicon_dictionary=lexicon_dictionary,
                                              feats_dictionary=feats_dictionary,
                                              symbolic_root=symbolic_root, symbolic_end=symbolic_end,
                                              dry_run=dry_run, env=env, word_lenghts_return=word_lenghts_return)

  bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
  data_variable = []
  ss = [0] * len(_buckets)
  ss1 = [0] * len(_buckets)

  for bucket_id in range(len(_buckets)):
    bucket_size = bucket_sizes[bucket_id]
    if bucket_size == 0:
      data_variable.append((1, 1))
      continue

    bucket_length = _buckets[bucket_id]
    MAX_TAGS = 6
    MAX_FEATS = 4
    #char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
    char_length = MAX_CHAR_LENGTH
    wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)

    if word_lenghts_return:
      masks_word_inputs = np.zeros([bucket_size, bucket_length, char_length], dtype=np.float32)

    c_lenghts = np.empty([bucket_size, bucket_length], dtype=np.int64)

    tags_inputs = np.empty([bucket_size, bucket_length, MAX_TAGS], dtype=np.int64)
    feats_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    if feats_lex_dic is not None:
      feats_inputs_lex = np.empty([bucket_size, bucket_length, MAX_FEATS], dtype=np.int64)
    else:
      feats_inputs_lex = None

    pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

    masks_inputs = np.zeros([bucket_size, bucket_length], dtype=np.float32)
    single_inputs = np.zeros([bucket_size, bucket_length], dtype=np.int64)

    lengths_inputs = np.empty(bucket_size, dtype=np.int64)
    
    order_inputs = np.empty(bucket_size, dtype=np.int64)
    raw_word_inputs = []

    raw_lines = []
    for i, inst in enumerate(data[bucket_id]):
      ss[bucket_id] += 1
      ss1[bucket_id] = bucket_length
      # default
      wids, cid_seqs, pids, hids, tids, orderid, word_raw, lines, lexicon_tags, feats, word_lenghts = inst
        #print("word-->", word_raw)
      #return char leng
      inst_size = len(wids)
      lengths_inputs[i] = inst_size
      order_inputs[i] = orderid
      raw_word_inputs.append(word_raw)
      # word ids
      wid_inputs[i, :inst_size] = wids
      if feats_dictionary is not None:
        feats_inputs[i, :inst_size] = feats
        feats_inputs[i, inst_size:] = PAD_ID_TAG
      wid_inputs[i, inst_size:] = PAD_ID_WORD
      if lexicon_dictionary is not None:
        for tag, tag_ids in enumerate(lexicon_tags):
          _max_len = min(len(tag_ids), MAX_TAGS)
          tags_inputs[i, tag, :_max_len] = tag_ids[:_max_len]
          tags_inputs[i, tag, len(tag_ids):] = PAD_ID_TAG
      if feats_lex_dic is not None:
        for word_id, word in enumerate(word_raw):
          _feats = feats_lex_dic.get(word,[0])
          _max_len = min(len(_feats), MAX_FEATS)
          feats_inputs_lex[i, word_id, :_max_len] = _feats[:_max_len]
          feats_inputs_lex[i, word_id, _max_len:] = PAD_ID_TAG
      for c, cids in enumerate(cid_seqs):
          cid_inputs[i, c, :len(cids)] = cids
          cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
          if word_lenghts_return:
            masks_word_inputs[i,c, : len(cids)] = 1.
      # filling empty character ids and lenghts for empty words
      cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
      if lexicon_dictionary is not None:
        tags_inputs[i, inst_size:, :] = PAD_ID_TAG
      if feats_lex_dic is not None:
        feats_inputs_lex[i, inst_size:, :] = PAD_ID_TAG
      #c_lenghts[i, inst_size:, :] = 0
      # pos ids
      pid_inputs[i, :inst_size] = pids
      pid_inputs[i, inst_size:] = PAD_ID_TAG
      # type ids
      tid_inputs[i, :inst_size] = tids
      tid_inputs[i, inst_size:] = PAD_ID_TAG
      # heads
      hid_inputs[i, :inst_size] = hids
      hid_inputs[i, inst_size:] = PAD_ID_TAG
      # masks
      masks_inputs[i, :inst_size] = 1.0
      for j, wid in enumerate(wids):
          if word_dictionary.is_singleton(wid):
              single_inputs[i, j] = 1
      raw_lines.append(lines)

    words = Variable(torch.from_numpy(wid_inputs), volatile=False)#requires_grad=False)
    chars = Variable(torch.from_numpy(cid_inputs), volatile=False)#requires_grad=False)
    if feats_dictionary is not None:
      feats = Variable(torch.from_numpy(feats_inputs), volatile=False)#requires_grad=False)
    else:
      feats = None

    if lexicon_dictionary is not None:
      tags = Variable(torch.from_numpy(tags_inputs),volatile=False)
    else:
      tags = None
    if feats_lex_dic is not None:
      feats_inputs_lex = Variable(torch.from_numpy(feats_inputs_lex), volatile=False)
    else:
      feats_inputs_lex = None

    pos = Variable(torch.from_numpy(pid_inputs), volatile=False)#requires_grad=False)
    heads = Variable(torch.from_numpy(hid_inputs), volatile=False)#requires_grad=False)
    types = Variable(torch.from_numpy(tid_inputs), volatile=False)#requires_grad=False)
    masks = Variable(torch.from_numpy(masks_inputs), volatile=False)#requires_grad=False)
    single = Variable(torch.from_numpy(single_inputs), volatile=False)#requires_grad=False)
    lengths = torch.from_numpy(lengths_inputs)
    if word_lenghts_return:
      masks_words = Variable(torch.from_numpy(masks_word_inputs), volatile=False)
    else:
      masks_words = None
    if use_gpu:
      words = words.cuda()
      if word_lenghts_return:
        masks_words = masks_words.cuda()
      chars = chars.cuda()
      pos = pos.cuda()
      heads = heads.cuda()
      types = types.cuda()
      masks = masks.cuda()
      single = single.cuda()
      lengths = lengths.cuda()
      if lexicon_dictionary is not None:
        tags = tags.cuda()
      if feats_dictionary is not None:
        feats = feats.cuda()
      if feats_lex_dic is not None:
        feats_inputs_lex = feats_inputs_lex.cuda()
    # default
    # CHAR LENGH NOT SUPPORTED !!
    data_variable.append((words, chars, pos, heads, types, masks, single, lengths, order_inputs, raw_word_inputs, tags, feats, feats_inputs_lex, masks_words, raw_lines))

  return data_variable, bucket_sizes, _buckets


def return_get_batch_variable(lexicon, features, features_input, masks_words ,tags, feats, feats_input, masks_word_input, index):
  ret_tag = tags[index] if lexicon else None
  ret_feats = feats[index] if features else None
  ret_feat_input = feats_input[index] if features_input else None
  ret_masks_words = masks_word_input[index] if masks_words else None
  return ret_tag, ret_feats, ret_feat_input, ret_masks_words


def get_batch_variable(data,
                       batch_size,
                       unk_replace=0.,
                       lexicon=False,
                       features=False,
                       features_input=False,
                       word_mask_return=False,
                       ):
  data_variable, bucket_sizes, _buckets = data
  total_size = float(sum(bucket_sizes))
  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

  # Choose a bucket according to data distribution. We pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number = np.random.random_sample()
  bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
  bucket_length = _buckets[bucket_id]
  # ALL cases
  words, chars, pos, heads, types, masks, single, lengths, _, _, tags, feats, feats_input, mask_words , _ = data_variable[bucket_id]
  bucket_size = bucket_sizes[bucket_id]
  batch_size = min(bucket_size, batch_size)
  index = torch.randperm(bucket_size).long()[:batch_size]
  if words.is_cuda:
    index = index.cuda()

  words = words[index]

  # discarding singleton
  if unk_replace:
    ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
    noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
    words = words * (ones - single[index] * noise)

  ret_tag, ret_feats, ret_feat_input, ret_mask_words = return_get_batch_variable(lexicon, features, features_input, word_mask_return, tags, feats, feats_input, mask_words,index)
  return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index], ret_tag, ret_feats, ret_feat_input, ret_mask_words



def return_iterate_batch_variable(lexicon, features, features_input, word_mask_return, tags, feats, feats_inputs_lex, word_mask_inputs, excerpt):
  ret_tag = tags[excerpt] if lexicon else None
  ret_features = feats[excerpt] if features else None
  ret_features_input = feats_inputs_lex[excerpt] if features_input else None
  ret_words_masks = word_mask_inputs[excerpt] if word_mask_return else None
  return ret_tag, ret_features, ret_features_input, ret_words_masks

def iterate_batch_variable(data, batch_size, unk_replace=0.,lexicon=False, features_input=False,features=False,word_mask_return=False):
  data_variable, bucket_sizes, _buckets = data
  bucket_indices = np.arange(len(_buckets))

  for bucket_id in bucket_indices:
    bucket_size = bucket_sizes[bucket_id]
    bucket_length = _buckets[bucket_id]
    if bucket_size == 0:
      continue
    # RAW_LINES
    #default
    words, chars, pos, heads, types, masks, single, lengths, order_ids, raw_word_inputs, tags, feats, feats_inputs_lex, mask_words ,raw_lines = data_variable[bucket_id]

    if unk_replace:
      ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
      noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
      words = words * (ones - single * noise)

    for start_idx in range(0, bucket_size, batch_size):
      excerpt = slice(start_idx, start_idx + batch_size)
      ret_tag, ret_features, ret_features_input, ret_words_masks = return_iterate_batch_variable(lexicon,features, features_input,word_mask_return,
                                                                                                   tags, feats, feats_inputs_lex,mask_words, excerpt)
      yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], masks[excerpt], lengths[excerpt], order_ids[excerpt],\
              raw_word_inputs[excerpt], ret_tag, ret_features, ret_features_input, ret_words_masks, raw_lines[excerpt]

def n_hot_tf(index_vec, n_label,use_gpu=False):
    if not use_gpu:
      one_hot = torch.eye(n_label)[index_vec.data,:]
    else:
      one_hot = torch.eye(n_label).cuda()
      one_hot = one_hot[index_vec.data,:]
    n_hot = torch.sum(one_hot, dim=-2)
    n_hot[n_hot>1]=0
    return n_hot




if __name__=="__main__":

  import uuid
  import encoding.utils as utils
  import os
  model_suff = "TEST"

  data_set = 'fr_sequoia'
  word_embedding_name = "fr.skip.forms.50.vectors"
  word_path = "../../parsing/data/ud/supdata/ud-2.0-baselinemodel-train-embeddings/"+word_embedding_name
  word_dict, word_dim = utils.load_embedding_dict("word2vec", word_path)

  model_name = str(uuid.uuid4())[:5]+"-"+model_suff
  model_path = "./models/"
  model_path = os.path.join(model_path, model_name)
  os.mkdir(model_path)
  alphabet_path = os.path.join(model_path, 'alphabets/')
  data_source_path = "../data/release-2.2-st-train-dev-data-NORMALIZED"
  train_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-train.conllu"
  dev_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
  test_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
  print("alphabet_path",alphabet_path)
  import pickle
  with open("../lexicons/Dic/French-Lefff-lex/lexicon_feats.pkl", "rb") as f:
    lexicon_feats = pickle.load(f)
  with open("../lexicons/Dic/French-Lefff-lex/lexicon_tags.pkl", "rb") as f:
    lexicon_tags = pickle.load(f)
    #print(lexicon_feats)
  word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_dict(alphabet_path,
                                                                          train_path,
                                                                          dev_path=dev_path,
                                                                          test_path=test_path,
                                                                          features=False,lexicon_feats_inst2id=lexicon_feats,lexicon_feats=True,
                                                                          word_embed_dict=word_dict,lexicon=lexicon_tags,
                                                                          dry_run=False, env="python3")
  pos, lex, feats , feats_lex = pos_alphabet
  #pos  = pos_alphabet
  #print(pos.instance2index)#, lex, feats.instance2index)
  print(max([a for ls in feats_lex for a in feats_lex[ls]]))
  print("FEATS LEX : ", len(feats_lex), len(word_alphabet.instance2index), len(lex))