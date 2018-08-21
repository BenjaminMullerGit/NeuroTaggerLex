# -*- coding: utf-8 -*-

import re
import codecs

class CoNLLWriter(object):
  def __init__(self, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, features_dictionary=None):
    self.__source_file = None
    self.__word_dictionary = word_dictionary
    self.__char_dictionary = char_dictionary
    self.__pos_dictionary = pos_dictionary
    self.__type_dictionary = type_dictionary
    self.__out_data = {}

  def start(self, file_path):
    self.__source_file = open(file_path, 'w') #codecs.open(file_path, 'w', 'utf-8')

  def close(self):
    self.__source_file.close()

  def extract_pos(self, p):
    if '$$$' not in p:
      return p, '_'
    items = p.split('$$$')
    return items[0], items[1]

  def store_buffer(self, word, pos, head, type, lengths, order_ids, raw_words, raw_lines, symbolic_root=False,
                   symbolic_end=False):
    batch_size, _ = word.shape
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0
    for i in range(batch_size):
      sent_tokens = []
      for j in range(start, lengths[i] - end):
        w = raw_words[i][j]
        p = self.__pos_dictionary.get_instance(pos[i, j])
        t = self.__type_dictionary.get_instance(type[i, j])
        h = head[i, j]
        upos, xpos = self.extract_pos(p)
        sent_tokens.append([w, upos, xpos, h, t])
      self.__out_data[order_ids[i]] = [sent_tokens, raw_lines[i]]

  def __is_multiple_root(self, sent_tokens):
    count_roots = 0
    for token in sent_tokens:
      if token[3] == 0:
        count_roots += 1
    return count_roots!=1

  def write_buffer(self):

    for seq_no in range(len(self.__out_data)):
      sent_tokens, raw_lines = self.__out_data[seq_no]
      cur_ti = 0
      for ud_tokens in raw_lines:
        idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc = ud_tokens
        if '-' not in idi and '.' not in idi:
          cur_model_tokens = sent_tokens[cur_ti]
          #head, typ = str(cur_model_tokens[1]), cur_model_tokens[2]
          upos = str(cur_model_tokens[1])
          cur_ti+=1

        self.__source_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(idi, form, lemma, upos, xpos, feats, head, deprel, deps, misc))
      self.__source_file.write("\n")

    def store_buffer_pos(self, word, pos, head, type, lengths, order_ids, raw_words, symbolic_root=False, symbolic_end=False):
      batch_size, _ = word.shape
      start = 1 if symbolic_root else 0
      end = 1 if symbolic_end else 0
      for i in range(batch_size):
        sent_tokens = []
        for j in range(start, lengths[i] - end):
          w = raw_words[i][j]
          p = self.__pos_dictionary.get_instance(pos[i, j])
          t = self.__type_dictionary.get_instance(type[i, j])
          h = head[i, j]
          upos, xpos = self.extract_pos(p)
          sent_tokens.append([w, upos, xpos, h, t])
        self.__out_data[order_ids[i]] = sent_tokens

  def write_pos(self, word, pos, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 0
        for i in range(batch_size):
            for j in range(start, lengths[i]):
                w = self.__word_dictionary.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_dictionary.get_instance(pos[i, j]).encode('utf-8')
                self.__source_file.write('%d\t%s\t_\t_\t%s\t_\t%s\t%s\n' % (j, w, p,"_","_"))
            self.__source_file.write('\n')


def arabic_post_process(file_path,suffix,verbose=False):
  file_write  = open(file_path+"."+suffix,'w')
  print("WRITING FILE TO {} ".format(file_path+"."+suffix))
  with codecs.open(file_path,'r') as f:
    i=0
    j=0
    W  = 0
    within_multi_word = 0
    for line in f:
      if line.startswith("#"):
        if line.startswith("# newpar"):
          file_write.write("\n")
        if line.startswith("# sent_id"):
          count_line=0
          if verbose:
            print("--------NEW SENT", line)
          file_write.write("\n")
        file_write.write(line)
      else:

        matching = re.match('^(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)$', line)
        j+=1
        if matching is not None:
          if verbose:
            print("MATCHING", matching.group(0))

          cat = re.match("^([0-9]+)-([0-9]+)",matching.group(1))
          # is it a multiword case ?
          # detect multiwords lines*

          if cat is not None:
            if verbose:
              print("MULTI ", cat.group(1), cat.group(2))
            len_multi_word = int(cat.group(2))-int(cat.group(1))
            if len_multi_word>1:
              if verbose:
                print("WARNING", matching.group(1), matching.group(2))
            within_multi_word = 1
            count_mutli_word_token = 0
            if verbose:
              print("SANITY-CHECK my counter {} real counter {}  multiword ? ".format(count_line, matching.group(1), "--"))
            file_write.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(str(count_line+1)+"-"+str(count_line+1+len_multi_word),
                                                                               matching.group(2), matching.group(3), matching.group(4), matching.group(5),
                                                                               matching.group(6), matching.group(7), matching.group(8), matching.group(9), matching.group(10)))

          # non multi-word line
          else:
            if verbose:
              print("+1", count_line)
            count_line += 1
            if verbose:
              print("SANITY-CHECK my counter {} real counter {} ".format(count_line, matching.group(1),"--"))
            if within_multi_word:
              count_mutli_word_token+=1
              if count_mutli_word_token==(len_multi_word+2):
                within_multi_word = 0
            if matching.group(4)=="X" and not within_multi_word:

              match_o = re.match("(و)(.*)",matching.group(2))

              if match_o is not None:
                i+=1
                file_write.write('{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n'.format(str(count_line)+"-"+str(count_line+1), matching.group(2)))
                file_write.write('{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n'.format(str(count_line), match_o.group(1)))
                file_write.write('{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n'.format(str(count_line+1), match_o.group(2)))
                count_line += 1
              else:
                file_write.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(str(count_line),matching.group(2), matching.group(3), matching.group(4), matching.group(5), matching.group(6),
                                                                                       matching.group(7), matching.group(8), matching.group(9),matching.group(10)))
            else:
              file_write.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(str(count_line),matching.group(2), matching.group(3), matching.group(4), matching.group(5),
                                                                                 matching.group(6), matching.group(7), matching.group(8), matching.group(9), matching.group(10)))


if __name__ == "__main__":

  test = False

  if test:
    with open("/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/data/tmp/demo_arabic/eval-ud.conllu","r") as f:
      for id, line in enumerate(f):
        if line.startswith("41-44"):
          print("OK", line)

  arabic_post_process("/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/data/tmp/demo_arabic/eval-ud.conllu", suffix="corr")