

def preprocess_parameters(lexicon,features, lexicon_feats, pos_alphabet):

    if lexicon is None and not features and not lexicon_feats:
        lexicon_alphabet = None
        features_dictionary = None
        num_feats_pred = None
        num_lexicon_feats = None
    elif lexicon is not None and not features and not lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        pos_alphabet = pos_alphabet[0]
        features_dictionary = None
        num_feats_pred = None
        num_lexicon_feats = None
    elif features and not lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        features_dictionary = pos_alphabet[2]
        pos_alphabet = pos_alphabet[0]
        num_feats_pred = len(features_dictionary.instance2index)+1
        num_lexicon_feats = None
    elif lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        features_dictionary = pos_alphabet[2]
        if features:
            num_feats_pred = len(features_dictionary.instance2index)+1
        else:
            num_feats_pred = None
        lexicon_feats_dictionary = pos_alphabet[3]
        num_lexicon_feats = max([a for ls in lexicon_feats_dictionary for a in lexicon_feats_dictionary[ls]])+1
        pos_alphabet = pos_alphabet[0]

    return lexicon_alphabet, features_dictionary, pos_alphabet, num_feats_pred, num_lexicon_feats
