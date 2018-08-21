from encoding.ioutils import DependencyInstance, Sentence
from encoding.constants import DIGIT_RE, MAX_CHAR_LENGTH, NUM_CHAR_PAD, ROOT, ROOT_CHAR, ROOT_POS, ROOT_TYPE, PAD, END, END_CHAR, END_POS,END_TYPE

FEATS_ORDERED = ["Case", "Gender", "Number", "PronType", "VerbForm", "Mood", "Voice"]


def preprocess_feats(feats):

  _feats = "_"

  if feats!="_":
    bag_of_feat = {}
    _feats = []
    for feat in feats.split('|'):
      morph_class, morph_val = feat.split('=')
      if morph_class in FEATS_ORDERED:
        bag_of_feat[morph_class] = morph_val
    for m_class in FEATS_ORDERED:
      if bag_of_feat.get(m_class):
       _feats.append(m_class+'='+bag_of_feat[m_class])
    if len(_feats) == 0:
      _feats = "_"
    else:
      _feats = "|".join(_feats)
  return _feats


class CoNLLReader(object):

  def __init__(self, file_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, feature_dictionary = None, lexicon_dictionary = None):
    self.__source_file = open(file_path, 'r')
    self.__word_dictionary = word_dictionary
    self.__char_dictionary = char_dictionary
    self.__pos_dictionary = pos_dictionary
    self.__type_dictionary = type_dictionary
    self.__lexicon_dictionary = lexicon_dictionary
    self.__feature_dictionary = feature_dictionary

  def close(self):
    self.__source_file.close()

  def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False, env="python2", word_len=False):

    assert env in ["python2", "python3"]

    line = self.__source_file.readline()

    # skip multiple blank lines.
    while len(line) > 0 and (len(line.strip()) == 0 or line.strip()[0]=='#'):
      line = self.__source_file.readline()
    if len(line) == 0:
      return None

    lines = []
    while len(line.strip()) > 0:
      line = line.strip()
      lines.append(line.split('\t'))
      line = self.__source_file.readline()

    length = len(lines)
    if length == 0:
      return None

    words = []
    word_ids = []
    char_seqs = []
    char_id_seqs = []
    postags = []
    pos_ids = []
    types = []
    type_ids = []
    heads = []

    if self.__lexicon_dictionary is not None:
      lexicon_tags = []
    else:
      lexicon_tags = None
    if self.__feature_dictionary is not None:
      feat_ids = []
    else:
      feat_ids = None
    if word_len:
      word_lenghts = []
    else:
      word_lenghts = None
    if symbolic_root:
      if word_len:
        word_lenghts.append(1)
      words.append(ROOT)
      word_ids.append(self.__word_dictionary.get_index(ROOT))
      char_seqs.append([ROOT_CHAR, ])
      char_id_seqs.append([self.__char_dictionary.get_index(ROOT_CHAR), ])
      postags.append(ROOT_POS)
      pos_ids.append(self.__pos_dictionary.get_index(ROOT_POS))
      types.append(ROOT_TYPE)
      type_ids.append(self.__type_dictionary.get_index(ROOT_TYPE))
      heads.append(0)
      if self.__lexicon_dictionary is not None:
        # we fill the same --> CHECK that index lexicon has something for
       lexicon_tags.append(self.__lexicon_dictionary.get(ROOT,[0]))
      if self.__feature_dictionary is not None:
        feat_ids.append(self.__word_dictionary.get_index(ROOT_POS))

    for tokens in lines:
      if '-' in tokens[0] or '.' in tokens[0]:
        continue
      chars = []
      char_ids = []
      for char in tokens[1]:
        chars.append(char)
        char_ids.append(self.__char_dictionary.get_index(char))
      if len(chars) > MAX_CHAR_LENGTH:
        chars = chars[:MAX_CHAR_LENGTH]
        char_ids = char_ids[:MAX_CHAR_LENGTH]

      char_seqs.append(chars)
      char_id_seqs.append(char_ids)
      if word_len:
        word_lenghts.append(len(chars))

      if env == "python3":
        word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      else:
        word = DIGIT_RE.sub(b"0", tokens[1])
      pos =  tokens[3] if tokens[4] == '_' else tokens[3]+'$$$'+tokens[4]
      head = tokens[6]
      type = tokens[7]

      words.append(tokens[1])
      word_ids.append(self.__word_dictionary.get_index(word))

      postags.append(pos)
      pos_ids.append(self.__pos_dictionary.get_index(pos))
      if self.__lexicon_dictionary is not None:
        lexicon_tags.append(self.__lexicon_dictionary.get(tokens[1], [0]))
      if self.__feature_dictionary is not None:
        feats = preprocess_feats(tokens[5])
        feat_ids.append(self.__feature_dictionary.get_index(feats))
      types.append(type)
      type_ids.append(self.__type_dictionary.get_index(type))

      heads.append(head)

    if symbolic_end:
      words.append(END)
      if word_len:
        word_lenghts.append(1)
      word_ids.append(self.__word_dictionary.get_index(END))
      char_seqs.append([END_CHAR, ])
      char_id_seqs.append([self.__char_dictionary.get_index(END_CHAR), ])
      postags.append(END_POS)
      pos_ids.append(self.__pos_dictionary.get_index(END_POS))
      types.append(END_TYPE)
      type_ids.append(self.__type_dictionary.get_index(END_TYPE))
      heads.append(0)
      if self.__lexicon_dictionary is not None:
        # !--> if no tokens ??
        lexicon_tags.append(self.__lexicon_dictionary.get(END))
      if self.__feature_dictionary is not None:
        feat_ids.append(self.__feature_dictionary.get_index(END_POS))

    return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, lines, word_lenght=word_lenghts, feats=feat_ids, lexicon_tags=lexicon_tags),
                              postags, pos_ids, heads, types, type_ids)



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
  from conllu_data import create_dict
  word_dictionary, char_dictionary, pos_dictionary, type_dictionary = create_dict(alphabet_path,
                                                                          train_path,
                                                                          dev_path=dev_path,
                                                                          test_path=test_path,
                                                                          features=False,lexicon_feats_inst2id=lexicon_feats,lexicon_feats=True,
                                                                          word_embed_dict=word_dict,lexicon=lexicon_tags,
                                                                          dry_run=False, env="python3")

  print(pos_dictionary[len(pos_dictionary)-1])
  reader = CoNLLReader(train_path, word_dictionary, char_dictionary, pos_dictionary[0], type_dictionary,feature_dictionary=None,
                       lexicon_dictionary=None)
  inst = reader.getNext(normalize_digits=False, symbolic_root=True, symbolic_end=False, env="python3",
                        word_len=True)
  sent = inst.sentence
  sent.words
  print(sent.word_ids, sent.char_seqs, sent.char_id_seqs, sent.lexicon_tags, sent.lines, sent.feats, sent.word_lengh)
  inst = reader.getNext(normalize_digits=False, symbolic_root=True, symbolic_end=False, env="python3",
                        word_len=True)
  sent = inst.sentence
  sent.words
  print(sent.word_ids, sent.char_seqs, sent.char_id_seqs, sent.lexicon_tags, sent.lines, sent.feats, sent.word_lengh)
  inst = reader.getNext(normalize_digits=False, symbolic_root=True, symbolic_end=False, env="python3",
                        word_len=False)
  sent = inst.sentence
  sent.words
  print(sent.word_ids, sent.char_seqs, sent.char_id_seqs, sent.lexicon_tags, sent.lines, sent.feats, sent.word_lengh)
