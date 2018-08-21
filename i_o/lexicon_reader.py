import codecs
import os
import pickle
from encoding.dictionary import Dictionary
#from constants import PAD_ID_MORPH, ROOT_ID_MORPH
PAD_ID_MORPH = 1
ROOT_ID_MORPH = 2

import re
import json
from i_o.conllu_data import preprocess_feats

class Lattice(object):
  def __init__(self, file_path_dir, file_name, dict_path,renaming_dic=None, only_pos=False,verbose=False):
    self.morph_token_dictionary = Dictionary('morph-token', default_value=True)
    self.morph_token_dictionary.add(PAD_ID_MORPH)
    self.morph_token_dictionary.add(ROOT_ID_MORPH)
    self.morph_class_dictionary = Dictionary('morph-class', default_value=True)
    self.morph_class_dictionary.add(PAD_ID_MORPH)
    self.morph_class_dictionary.add(ROOT_ID_MORPH)
    self.morph_full_feats = Dictionary('morph-full', default_value=True)
    self.morph_value_dictionary = Dictionary('morph-value', default_value=True)
    self.morph_full_feats_filtered = Dictionary('feat-restricted', default_value=True)
    self.morph_full_feats_filtered.add(PAD_ID_MORPH)
    self.morph_full_feats_filtered.add(ROOT_ID_MORPH)
    self.morph_value_dictionary.add(PAD_ID_MORPH)
    self.morph_value_dictionary.add(ROOT_ID_MORPH)

    file_path = file_path_dir + "UDLex_"+file_name+".conllul"

    if renaming_dic is not None:
      file_name = renaming_dic[file_name.split("-")[0]]

    if not os.path.isdir(dict_path+file_name+"-lex"):

      os.mkdir(dict_path+file_name+"-lex")
      print("Dir {} created ".format(dict_path+file_name))
    path = dict_path+file_name+"-lex"
    self.wordpos2feats = {}
    self.word2tags = {}

    self.word2feats = {}

    with codecs.open(file_path, 'r', 'utf-8') as f:
      for line in f:
        content = line.strip().split('\t')
        morph_feat = content[6]

        if self.word2tags.get(content[2], False):
          if verbose:
            print("Appending-->", content[2], content[4])
          self.word2tags[content[2]].append(content[4])
        else:
          if verbose:
            print("NEW-->", content[2], content[4])
          self.word2tags[content[2]] = [content[4]]

        if not only_pos:
          if morph_feat != '_':
            token = content[2]
            morph_token = content[2]+'$$$'+content[4]
            self.morph_token_dictionary.add(morph_token)
            word_feats = []
            feat_filtered = preprocess_feats(morph_feat)
            self.word2feats[content[2]] = feat_filtered
            for feat in feat_filtered.split('|'):
              if feat != "_":
                self.morph_full_feats_filtered.add(feat)
            for feat in morph_feat.split('|'):
              morph_class, morph_val = feat.split('=')
              self.morph_full_feats.add(feat)
              self.morph_class_dictionary.add(morph_class)
              self.morph_value_dictionary.add(morph_class+'='+morph_val)
              word_feats.append([self.morph_class_dictionary.get_index(morph_class), self.morph_value_dictionary.get_index(morph_class+'='+morph_val)])

            if morph_token not in self.wordpos2feats:
              self.wordpos2feats[morph_token] = []
            self.wordpos2feats[morph_token] += word_feats

            if token not in self.wordpos2feats:
              self.wordpos2feats[token] = []
            self.wordpos2feats[token] += word_feats

    print('lexicon: #tokens=%d; #class=%d; #value=%d;'%(self.morph_token_dictionary.size(), self.morph_class_dictionary.size(), self.morph_value_dictionary.size()))
    for word in self.word2tags:
      self.word2tags[word] = list(set(self.word2tags[word]))
    self.word2feats_ids = {}
    self.word2feats_ids["_ROOT"] = [ROOT_ID_MORPH]
    #self.word2feats_ids["_ROOT"] = ROOT_ID_MORPH
    for word in self.word2feats:
      if self.word2feats_ids.get(word,None) is None:
        self.word2feats_ids[word] = []
        feat_filtered = self.word2feats[word]
        #print("FEAT FILTERED", feat_filtered)
        for feat in feat_filtered.split('|'):
          self.word2feats_ids[word].append(self.morph_full_feats_filtered.get_index(feat))
          #print("appending ", feat, self.morph_full_feats_filtered.get_index(feat))


      self.word2tags[word] = list(set(self.word2tags[word]))
    with open(path+"/lexicon_feats.pkl","wb") as f:
      pickle.dump(self.word2feats_ids, f)
    with open(path+"/lexicon_tags.pkl","wb") as f:
      pickle.dump(self.word2tags, f)
    self.morph_token_dictionary.save(path)
    self.morph_class_dictionary.save(path)
    self.morph_full_feats.save(path)
    self.morph_value_dictionary.save(path)
    self.morph_full_feats_filtered.save(path)
    print(self.word2feats)
    print("Lang {} Everything saved to {} ".format(file_name,path))
    #print(self.morph_full_feats.instance2index)
    #print(self.word2tags)

if __name__=="__main__":


  #with open("../../data/ls_match_2.txt","r") as f:
  #  dic = {}
  #  for line in f:
  #   cat = re.search("(.*) (.*)",line)
  #   dic[cat.group(1)] = cat.group(2)
  #with open("../../data/dic_lang_link.json","w") as f:
  #  json.dump(dic,f)
  dic_2 = {}
  if False:
    with open("../../data/dic_lang_link.json","r") as f:
      dic = json.load(f)
    for key in dic:
      print("--K", key)
      cat_key = re.search("(.*)-.*",key).group(1)
      if dic[key] !="NONE":
        cat_value = re.search("(.*)_.*",dic[key]).group(1)
      else:
        cat_value = "NONE"
      if dic_2.get(cat_key, True):
        dic_2[cat_key] = cat_value
      elif dic_2.get(cat_key) == "NONE":
        dic_2[cat_key] = cat_value
      elif cat_value == "NONE":
        pass
      if cat_key=="English":
        print("-->", dic_2[cat_key], cat_value)

      print(key, dic[key], cat_key, cat_value)
    print(dic_2)
  if False:
    with open("../../lexicons/ls.txt","r") as f:
      i=0
      for line in f:
        print(line)
        if i!=0:
          print(line)
          cat = re.search("UDLex_((.*)-.*)\..*", line)
          print("ok")
          print(cat.group(1),cat.group(2))
          lat = Lattice("../../lexicons/UDLexicons.0.2/", cat.group(1),
                      dict_path="../../lexicons/Dic/", only_pos=False)
        i+=1

  #with open("../../lexicons/Dic/Kazakh-ApertiumMA-lex/lexicon_feats.pkl","rb") as f:
  #    lexicon_feats = pickle.load(f)
  #    print(lexicon_feats)

      #lat.word2tags
  PREPROCESS_LEXICONS = False
  if PREPROCESS_LEXICONS:

    label2lang_id = json.load(open("../lex2lang_id.json","r"))
    errors = []
    for _set in open("../../lexicons/UDLexicons.0.2/ls_.txt","r"):
      print("Processing {} set".format(_set[:-1]))
      try:
        lat = Lattice("../../lexicons/UDLexicons.0.2/", _set[:-1],
                 dict_path="../../lexicons/Lexicon_normalized_2/", only_pos=False, renaming_dic=label2lang_id)
      except KeyError as e:
        print(" ERROR with, ", e)
        errors.append(e)
      except ValueError as f:
        print("ERROR VALUE", f)
        errors.append(f)
        print("--> errors  : ", errors)
  label2lang_id = {}


  GENERATE_DICTIONARY = False
  if PREPROCESS_LEXICONS:
    with open("../tb2wv.txt",'r') as f:
      for line in f :
        split_1 = line.split(" ")
        print("BEFORE ", split_1)
        if len(split_1)>2:
          del split_1[1]
        if split_1[0] != "Arabic-PADT":
          split_1[1] = split_1[1][:-1]
        split_1[0] = split_1[0].split("-")[0]
        label2lang_id[split_1[0]] = split_1[1]
    label2lang_id = json.dump(label2lang_id,open("../lex2lang_id.json","w"))

      #with open("../../lexicons/Lexicons_normalized"+" /ar-lex/lexicon_tags.pkl","rb") as f:
    #  lexicon = pickle.load(f)
    #  print(lexicon)
  WRITE_LANG_CARACTERISTICS_LIST = True
  if WRITE_LANG_CARACTERISTICS_LIST:
    json.dump({},open("../../list/dic_char_2.json",'w'),)

    list_lex = []
    list_ud = []
    list_lang = []
    for lex in open("../../list/ls_lexicons.txt",'r'):
        lexicon = lex[:-5]
        print("LEXICONS " , lexicon)
        list_lex.append(lexicon)
    for fold in open("../../list/list_data_sets_ud_direct.txt",'r'):
        folder = fold[3:-1]
        print("DIRECT FOLDER ", folder)
        list_ud.append(folder)
    for lang in open("../../list/list_lang_id.txt",'r'):
        lang_id = lang[:-1]
        print("LANG", lang_id )
        list_lang.append(lang_id )
    if True:
      print(list_lex)
      for file in list_ud:
        lang = file[:2]
        d = json.load(open("../../list/dic_char_2.json",'r'))

        d[file] =  {"lang": lang,
                    "lexicon_tags": int(lang in list_lex),
                    "lexicon_feats":int(lang in list_lex)
                    }
        print(d)
        json.dump(d,open("../../list/dic_char_2.json",'w'),)
