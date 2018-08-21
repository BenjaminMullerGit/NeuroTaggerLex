# -*- coding: utf-8 -*-

"""
Evaluation
"""
import os
import sys
import argparse
import pickle
from model import NeuroTagger
from evaluation.evaluation_tagging import evaluate
import torch
from i_o.conllu_writer import CoNLLWriter
import i_o.conllu_data as conllu_data
import json
from args.args_neurotagger import score_args
UNK_ID = 0
BUCKETS_TEST = [500]


RUN_LOCALLY = False
PROJECT_PATH = os.environ["PROJECT_PATH"]
REPORT_PATH = os.environ["REPORT_PATH"]


def score():

    by_hand = False
    use_gpu = torch.cuda.is_available()


    print("INFO scoring : 'gpu found' is {} ".format(use_gpu))

    args = score_args()

    dev_eval = True

    if args.multilingual=="1":
        assert args.data_set_model is not None
        data_set_model = args.data_set_model
        dev_eval = False
    else:
        data_set_model = args.data_set
        
    writing_logs = True
    # Arguments 
    data_set = args.data_set
    run_id = args.run_id
    data_source_path = args.data_source_path_train
    model_id = args.model_id
    #lexicons_path = args.lexicons_path
    log_path = os.path.join(PROJECT_PATH, "models/log_run-"+run_id+"_rid-"+model_id+"_id.json")
    
    # LOADING PARAMETERS
    with open(log_path,"r") as f:
        logs = json.load(f)
    model_name = run_id+"_rid-"+model_id+"_id"+"-"+ data_set_model+"_data"
    model_path = os.path.join(PROJECT_PATH,"models", model_name)
    path_json = model_path +"/"+ model_name+'-arguments.json'
    with open(path_json,"r") as f:
        arg_json = json.load(f)
    print("ARGS LOADED", arg_json)
    char_dim= arg_json["char_dim"]
    # NB word dim not saved  careful with them
    if arg_json.get("word_dim", None) is None:
        word_dim = 300
    else:
        word_dim = arg_json.get("word_dim")

    if arg_json.get("attention_char", None) is not None:
        attention_char = arg_json["attention_char"]
    else:
        attention_char = False

    lexicon = arg_json.get("lexicon", None)
    if lexicon is not None and lexicon != "No":
        if args.multilingual is None or args.multilingual == "0":
            #lexicon =  lexicons_path +"/"+ data_set[:2]+"-lex/lexicon_tags.pkl"
            with open(lexicon,"rb") as f:
                print("INFO : lexicon loaded from ", lexicon)
                lexicon = pickle.load(f)
        elif args.multilingual == "1":
            assert args.lexicon_path is not None
            with open(args.lexicons_path,"rb") as f:
                print("other LEXICON LOADING from ", args.lexicon_path)
                lexicon = pickle.load(f)
        if not arg_json.get("pos_lexicon_dim",True):
            pos_lexicon_dim = arg_json.get("pos_lexicon_dim")
        else:
            pos_lexicon_dim = 300
            print("WARNING : pos_lexicon_dim setted by hand to {} ".format(pos_lexicon_dim))
    else:
        pos_lexicon_dim = 2
    lexicon_bool = False if (lexicon is None or lexicon == "No") else True
    if lexicon == "No":
        lexicon = None

    lexicon_feats = arg_json.get("lexicon_feats", None)
    if lexicon_feats:
        path_lexicon_feats = arg_json.get("path_lexicon_feats",None)
        assert path_lexicon_feats is not None
        #path_lexicon_feats = lexicons_path +"/"+ data_set[:2]+"-lex/lexicon_feats.pkl"
        print("--> starting Loading Lexicon features ", path_lexicon_feats)
        with open(path_lexicon_feats,"rb") as f:
            lexicon_feats_inst2id = pickle.load(f)
        print("LOADING DIC2iD")
        feature_dim_lexicon = arg_json["feature_dim_lexicon"]
    else:
        lexicon_feats_inst2id=None
        feature_dim_lexicon = 0

    if arg_json.get("features", None) is not None:
        features = arg_json["features"]
        feats_space = arg_json["feats_space"]
        num_feats_pred = arg_json["num_feats_pred"]
    else:
        features = False
        feats_space = 0
        num_feats_pred = 0
    if arg_json.get("lexicon_mode", None) is not None:
        lexicon_mode = arg_json["lexicon_mode"]
        if lexicon_mode == "n-hot":
            use_gpu = False
            print("INFO : gpu offset for n-hot reasons")
    else:
        lexicon_mode = "continuous"
    if arg_json.get("char_mode", None) is not None:
        char_mode = arg_json["char_mode"]
    else:
        char_mode = "CNN"
    if char_mode=="RNN":
        word_lenghts_return= True
    else:
        word_lenghts_return= False



    n_hidden_layers = arg_json["n_hidden_layers"]
    pos_space = arg_json["pos_space"]
    hidden_size = arg_json["hidden_size"]
    num_filters = arg_json["num_filters"]
    agg_mode = arg_json["agg_mode"]
    # add lexicon + features (but treats case where it's not )
    #model_full_path = logs[model_id]["model_full_path"]
    # !TIRA CHANGE
    #model_path = logs[model_id]["model_path"] 
    data_format = "CONLLU"
    env = "python3"

    # todo : should be removed
    if by_hand:
        data_source_path = "../data/udpipe-trained/direct"

    
    batch_size = 30

    assert data_format == "CONLLU"
    if by_hand:
        model_path = PROJECT_PATH+"/models/"
        model_name = "0afc5-fr_sequoia-FR-BATCH_24-MAX_7"
        model_id = model_name
        model_path = os.path.join(model_path, model_name)
    alphabet_path = os.path.join(model_path, 'alphabets/')

    if dev_eval:
        #train_path = data_source_path+"/"+data_set+"/model-train.conllu"
        #dev_path = data_source_path+"/"+data_set+"/model-dev.conllu"
        #train_path = data_source_path+"/ud-treebanks-v2.2.renamed/ud-"+data_set+"/"+data_set+"-ud-train.conllu"
        #dev_path = data_source_path+"/ud-treebanks-v2.2.renamed/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
        train_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-train.conllu"
        dev_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
    else:
        dev_path = ""
        train_path = ""

    if args.test_file_name_custom is None:
        test_file_name = data_set+"-ud-test-pred.conllu"
        test_path = os.path.join(args.data_source_path_test, test_file_name)
    else:
        test_path = os.path.join(args.data_source_path_test, args.test_file_name_custom)
    #test_path = data_source_path+"/ud-"+data_set+"/eval-ud.conllu"


    conll_data = conllu_data
    create_dict = conll_data.create_dict
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(alphabet_path)
        print("INFO : model dir created {} and alphabet dir {}".format(model_path, alphabet_path))
    # NB ! be consistent in the way you defined, load and expand your dicitonnaries !ç!

    print("STARTING LOADED")
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_dict(alphabet_path,
                                                                            "", dev_path="", test_path=test_path,
                                                                            env=env,
                                                                            word_embed_dict=None,
                                                                            lexicon=lexicon,
                                                                            features=features,
                                                                            lexicon_feats=lexicon_feats,
                                                                            lexicon_feats_inst2id=lexicon_feats_inst2id,
                                                                            expand=lexicon_bool,
                                                                            dry_run=False)
    if lexicon is None and not features and not lexicon_feats:
        lexicon_alphabet = None
        features_dictionary = None
        lexicon_feats_dictionary = None
        num_lexicon_feats = 0
    elif lexicon is not None and not features and not lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        pos_alphabet = pos_alphabet[0]
        features_dictionary = None
        lexicon_feats_dictionary = None
        num_lexicon_feats = 0
    elif features and not lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        features_dictionary = pos_alphabet[2]
        pos_alphabet = pos_alphabet[0]
        lexicon_feats_dictionary = None
        num_lexicon_feats = 0
    elif lexicon_feats:
        lexicon_alphabet = pos_alphabet[1]
        features_dictionary = pos_alphabet[2]
        if features:
            num_feats_pred = len(features_dictionary.instance2index)+1
        lexicon_feats_dictionary = pos_alphabet[3]
        num_lexicon_feats = max([a for ls in lexicon_feats_dictionary for a in lexicon_feats_dictionary[ls]])+1
        pos_alphabet = pos_alphabet[0]

    num_words, num_chars, num_pos, num_types = word_alphabet.size(), char_alphabet.size(),\
                                                pos_alphabet.size(),  type_alphabet.size()

    conll_data = conllu_data

    if dev_eval:
        data_dev = conll_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,type_alphabet,
                                                    symbolic_end=False, env=env,
                                                    lexicon_dictionary=lexicon_alphabet, feats_lex_dic=lexicon_feats_dictionary,
                                                    feats_dictionary=features_dictionary,
                                                    symbolic_root=True,
                                                    word_lenghts_return=word_lenghts_return,
                                                    use_gpu=use_gpu,
                                                    )
        data_train = conll_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,type_alphabet,
                                                      symbolic_end=False,env=env,feats_dictionary=features_dictionary, feats_lex_dic=lexicon_feats_dictionary,
                                                      symbolic_root=True,lexicon_dictionary=lexicon_alphabet,
                                                      word_lenghts_return=word_lenghts_return,
                                                      use_gpu=use_gpu,
                                                      )
    data_test = conll_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet,type_alphabet,
                                                 symbolic_end=False,env=env,
                                                 lexicon_dictionary=lexicon_alphabet, feats_lex_dic=lexicon_feats_dictionary,
                                                 feats_dictionary=features_dictionary,
                                                 word_lenghts_return=word_lenghts_return,
                                                 symbolic_root=True,
                                                 use_gpu=use_gpu)
    dev_pos_corr_total = 0.
    dev_pos_complete_match = 0.
    dev_total = 0.
    n_total_inst_dev = batch_size# ?
    train_pos_corr_total = 0
    train_pos_complete_match = 0
    train_total = 0
    n_total_inst_train = 0

    if dev_eval:
        pred_writer_dev = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, features_dictionary)
        gold_writer_dev = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet,features_dictionary)
        pred_writer_train = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet,features_dictionary)
        gold_writer_train = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet,features_dictionary)
    pred_writer_test = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet,features_dictionary)
    gold_writer_test = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet,features_dictionary)
    if dev_eval:
        dev_file_pred = '{}-FINAL-pred_dev.conllu'.format(str(model_id))
        train_file_pred = '{}-FINAL-pred_train.conllu'.format(str(model_id))
        pred_filename_dev_path = model_path+'/'+dev_file_pred
        gold_filename_dev_path = model_path+'/{}-FINAL-gold_dev.conllu'.format(str(model_id))
        pred_filename_train_path = model_path+'/'+train_file_pred
        gold_filename_train_path = model_path+'/{}-FINAL-gold_train.conllu'.format(str(model_id))

    test_file_pred = '{}-SCORE-pred_test'.format(str(model_id))
    pred_filename_test_path = model_path+'/'+test_file_pred
    gold_filename_test_path = model_path+'/{}-SCORE-gold_test'.format(str(model_id))
    pred_writer_test.start(pred_filename_test_path)
    gold_writer_test.start(gold_filename_test_path)
    if dev_eval:
        _model_path = "/home/benjamin/parsing/NeuroTagger"+model_path[1:]
        with open(os.path.join(REPORT_PATH, run_id+"_run_id-tag_paths.txt"), "a") as f:
            f.write("{} {} {} {} {}\n".format(data_set, _model_path, train_file_pred, dev_file_pred, test_file_pred))


    if dev_eval:
        gold_writer_dev .start(gold_filename_dev_path)
        pred_writer_dev.start(pred_filename_dev_path)
        gold_writer_train.start(gold_filename_train_path)
        pred_writer_train.start(pred_filename_train_path)
    model = NeuroTagger(word_dim=word_dim,
                        char=True,
                        pos=False,
                        pos_space=pos_space,
                        embedd_word=None,
                        attention_char=attention_char,
                        num_words=num_words,char_mode=char_mode,
                        feat_pred=features, num_feats_pred=num_feats_pred,
                        feats_space=feats_space,
                        char_dim=char_dim,
                        num_chars=num_chars, 
                        num_pos=num_pos,
                        num_filters=num_filters,
                        pos_lexicon_dim=pos_lexicon_dim,
                        feature_dim_lexicon=feature_dim_lexicon,
                        features_lexicon=lexicon_feats,
                        num_feats_lexicon=num_lexicon_feats,
                        kernel_size=3,
                        rnn_mode="LSTM",
                        agg_mode=agg_mode,
                        lexicon=lexicon_bool,
                        hidden_size=hidden_size,
                        num_layers=n_hidden_layers,
                        lexicon_mode=lexicon_mode,
                        p_in=0.33, p_out=0.33, p_rnn=(0.5, 0.5),
                        p_class=0.3)

    model_full_path = os.path.join(model_path, "model")

    if use_gpu:
        model.load_state_dict(torch.load(model_full_path))
    else:
        model.load_state_dict(torch.load(model_full_path, map_location=lambda storage, loc: storage)) 
    print("LOADING DONE ")

    if use_gpu:
        model.cuda()

    model.eval()

    if dev_eval:
        args_train_data = {"data": data_train, "batch_size": batch_size,"lexicon":lexicon_bool,"word_mask_return":word_lenghts_return,
                            "features":features,"features_input":lexicon_feats}
        #PREDICTING TRAIN
        for batch in conll_data.iterate_batch_variable(**args_train_data):

            word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks, lines_gold = batch

            heads_gold = heads_gold.data.cpu().numpy()
            types_gold = types_gold.data.cpu().numpy()

            pos_pred, _ = model.predict(input_word=word, input_char=char,lexicon_tags=tags, mask=masks, 
                                        words_masks=words_masks,
                                        length=lengths, feats_lexicon=feats_input)
            # real data
            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            pos_pred = pos_pred.data.cpu().numpy()

            (pos_corr_train, total_train, pos_complete_match_train), _, batch_size = evaluate(words=word, pos_pred=pos_pred,
                                                                         pos=pos, lengths=lengths)
            train_pos_corr_total += pos_corr_train
            train_pos_complete_match += pos_complete_match_train
            train_total += total_train
            n_total_inst_train += batch_size


            pred_writer_train.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,
                                         raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)
            gold_writer_train.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words,
                                         raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)
        pred_writer_train.write_buffer()
        gold_writer_train.write_buffer()

        pred_writer_train.close()
        gold_writer_train.close()
        print("INFO : PERFORMANCE ON TRAIN ")
        print('-- Correct prediction {}, out of {} predictions/true value,  Correct % : {}, Correct Complete match %: {}'.format(
         train_pos_corr_total, train_total,
         train_pos_corr_total* 100 / train_total,
         train_pos_complete_match * 100 / n_total_inst_train))

        args_dev = {"data": data_dev, "batch_size": batch_size, "lexicon":lexicon_bool,"word_mask_return":word_lenghts_return,
                     "features":features,"features_input":lexicon_feats}


        #PREDICTING DEV
        for batch in conll_data.iterate_batch_variable(**args_dev):
            word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks, lines_gold = batch

            heads_gold = heads_gold.data.cpu().numpy()
            types_gold = types_gold.data.cpu().numpy()

            pos_pred, _ = model.predict(input_word=word, input_char=char,lexicon_tags=tags,feats_lexicon=feats_input,
                                        words_masks=words_masks,
                                        mask=masks, length=lengths,
                                        use_gpu=use_gpu)
            # real data
            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            pos_pred = pos_pred.data.cpu().numpy()

            (pos_corr, total, pos_complete_match),_, batch_size = evaluate(words=word, pos_pred=pos_pred,
                                                                           pos=pos, lengths=lengths)

            dev_pos_corr_total += pos_corr
            dev_pos_complete_match += pos_complete_match
            dev_total += total
            n_total_inst_dev += batch_size



            pred_writer_dev.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,
                                         raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)
            gold_writer_dev.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words,
                                         raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)

        pred_writer_dev.write_buffer()
        gold_writer_dev.write_buffer()

        pred_writer_dev.close()
        gold_writer_dev.close()
        print("INFO : PERFORMANCE ON DEV")
        print('-- Correct prediction {}, out of {} predictions/true value,  Correct % : {}, Correct Complete match %: {}'.format(
         dev_pos_corr_total, dev_total,
         dev_pos_corr_total* 100 / dev_total,
         dev_pos_complete_match * 100 / n_total_inst_dev))

    test_pos_corr_total = 0
    test_pos_complete_match = 0
    test_total = 0
    n_total_inst_test = 0

    args_test = {"data": data_test, "batch_size": batch_size,"lexicon":lexicon_bool,"word_mask_return":word_lenghts_return,
                    "features":features,"features_input":lexicon_feats}

    # EVALUATING ON TEST 
    for batch in conll_data.iterate_batch_variable(**args_test):

        word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks, lines_gold = batch

        heads_gold = heads_gold.data.cpu().numpy()
        types_gold = types_gold.data.cpu().numpy()

        pos_pred, _ = model.predict(input_word=word, input_char=char,lexicon_tags=tags,feats_lexicon=feats_input,
                                    words_masks=words_masks,
                                    use_gpu=use_gpu,
                                    mask=masks, length=lengths)

        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        lengths = lengths.cpu().numpy()
        pos_pred = pos_pred.data.cpu().numpy()

        # score
        (pos_corr, total, pos_complete_match), _ ,batch_size = evaluate(words=word, pos_pred=pos_pred,
                                                                     pos=pos, lengths=lengths)
        
        test_pos_corr_total += pos_corr
        test_pos_complete_match += pos_complete_match
        test_total += total
        n_total_inst_test += batch_size

        pred_writer_test.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,raw_lines=lines_gold,
                                     symbolic_root=True, symbolic_end=False)
        gold_writer_test.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words,raw_lines=lines_gold,
                                     symbolic_root=True, symbolic_end=False)


    pred_writer_test.write_buffer()
    gold_writer_test.write_buffer()

    pred_writer_test.close()
    gold_writer_test.close()
    print("PREDICTION WRITTEN to {}".format(pred_filename_test_path))
    if dev_eval:
        print("TRAINING AND DEV prediction  WRITTEN to {} and {} ".format(pred_filename_train_path,pred_filename_dev_path))
    print("INFO : PERFORMANCE ON TEST ")
    print('--> Correct prediction {} out of {} predictions/true value,  Correct {} %, Correct Complete match {} % '.format(
           test_pos_corr_total, test_total,
           test_pos_corr_total* 100 / test_total,
           test_pos_complete_match * 100 / n_total_inst_test))

    with open(log_path, "w") as f:
        logs[model_id]["pred_filename_path"] = pred_filename_test_path
        logs[model_id]["TRAIN_PRED_GOLD_TOK"] = None
        logs[model_id]["DEV_PRED_GOLD_TOK"] = None
        if dev_eval:
            logs[model_id]["TRAIN_PRED_GOLD_TOK"] = pred_filename_train_path
            logs[model_id]["DEV_PRED_GOLD_TOK"] = pred_filename_dev_path
        json.dump(logs, f)


if __name__=="__main__":
    score()
