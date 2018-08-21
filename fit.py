# -*- coding: utf-8 -*-

import uuid
import json
import os
import time
import torch
from torch.optim import Adam
from encoding.predict_toolkit import construct_word_embedding_table
import encoding.utils as utils

from i_o.conllu_writer import CoNLLWriter
import i_o.conllu_data as conllu_data
from args.tools import preprocess_parameters


from torch.nn.utils import clip_grad_norm
from model import NeuroTagger
from evaluation.evaluation_tagging import evaluate
from evaluation.reporting import plot_curves
import pickle
from args.args_neurotagger import fit_args
from encoding.constants import UNK_ID
from nn_tools.optimize import generate_optimizer

RUN_LOCALLY = False
CONTROL_LEARNING = False

WORD_EMBEDDING = "word2vec"
PROJECT_PATH = os.environ["PROJECT_PATH"]
REPORT_PATH = os.environ["REPORT_PATH"]
WORD_EMBEDDING_PATH = os.environ["WORD_EMBEDDING_PATH"]

# TODO : split this script
# - should have one funciton for parameters definitions, reporting and info
# - one for defining dictionaries : almost done
# - one for defining iterators on the data : train dev and test
# - one for defining model : done
# -

def main():
    # run settings (hardcoded because not meant to change)
    score = True
    reporting = True
    environment = "python3"
    verbose_extreme = False
    debug = False
    verbose = True
    writing_logs = True
    word_embedding = WORD_EMBEDDING

    required = not RUN_LOCALLY
    args = fit_args(required)
    # data
    data_set = args.data_set
    # environement and model tracking set up
    run_id = args.run_id
    data_source_path = args.data_source_path
    model_id = args.model_id
    lexicons_path = args.lexicons_path

    # model id
    if not RUN_LOCALLY:
        use_gpu = torch.cuda.is_available()
        if verbose:
            print("INFO : hardware 'gpu found' is  {} ".format(use_gpu))
        use_lexicon = int(args.use_lexicon)
        lexicon_feats = int(args.lexicon_feats)
        prerun = int(args.prerun)
        model_full_id = run_id+"_rid-"+model_id+"_id"
    else:
        data_set = "da_ddt"
        use_gpu = False
        model_full_id = str(uuid.uuid4())[:5]
        prerun = 1
        use_lexicon = 0
        lexicon_feats = False

    model_suff = data_set+"_data"
    model_name = model_full_id+"-"+model_suff
    model_path = os.path.join(PROJECT_PATH,"models/")
    model_path = os.path.join(model_path, model_name)
    alphabet_path = os.path.join(model_path, 'alphabets/')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        print("INFO : model dir created {} and alphabet dir {}".format(model_path, alphabet_path))


    # model hyperparameters
    # TODO : pass some of these parameters in arguments
    # architecture parameters
    hidden_size = int(args.hidden_size)
    attention_char = int(args.attention_char)
    num_epoch = int(args.num_epoch)
    batch_size = int(args.batch_size)
    lexicon_mode = args.lexicon_mode#"continuous"
    agg_mode = args.agg_mode#"SUM"



    char_mode = args.char_mode
    char_dim = 100
    # num_filter has two meaning :
    # if char_mode=="CNN" is the size of the cnn layer otherwise it's the recurrent cell hidden state dimension
    num_filters = 400
    feature_dim_lexicon = 300
    pos_lexicon_dim = 300
    unk_replace = 0.
    n_hidden_layers = 2
    feats_space = 100
    pos_space = 100
    features = True
    freeze_word_embedding = False
    multilingual = False
    random_init = False
    # optimization parameters
    lr = 0.001
    clip = 5.0
    decay_rate = 0.85
    eps = 1e-8
    betas = (0.9, 0.9)
    gamma = 0.
    schedule = 12.
    max_decay = 6
    double_schedule_decay = 5
    p_in = 0.1
    p_out = 0.1
    p_rnn = (0., 0.5)
    p_class = 0.1

    # prerun is meant for quick full pipeline testing : based  on small batch size,
    # random init for not having to load external word embedding, and 1 epoch only
    if prerun:
        print("--------------PRE-RUN----------")
        random_init = True
        batch_size = 50
        num_epoch = 1
        print("INFO : num_epochs set to {} and batch_size to {} ".format(num_epoch, batch_size))
    else:
        print("--------------REAL-RUN----------")

    # lexicon set up
    # TODO PATH should be cleanign and put in an appropriate file
    if not use_lexicon and not RUN_LOCALLY:
        lexicon = None
        path_lexicon  = "No"
    elif not RUN_LOCALLY:
        path_lexicon = os.path.join(lexicons_path, data_set[:2]+"-lex/lexicon_tags.pkl")
        with open(path_lexicon,"rb") as f:
            lexicon = pickle.load(f)
            print("INFO LEXICON loaded from {} ".format(path_lexicon))
    if not lexicon_feats and not RUN_LOCALLY:
        lexicon_feats_inst2id = None
        path_lexicon_feats = "No"
        feature_dim_lexicon = 0
    elif lexicon_feats and not RUN_LOCALLY:
        path_lexicon_feats = os.path.join(lexicons_path, data_set[:2]+"-lex/lexicon_feats.pkl")
        with open(path_lexicon_feats,"rb") as f:
            lexicon_feats_inst2id = pickle.load(f)
            print("INFO LEXICON FEATURES loaded from {} ".format(path_lexicon_feats))
    # TODO : same comment as above about paths
    # defining word embedding and lexicon paths and name
    if not RUN_LOCALLY:
        w_code = data_set[:2]
        if args.word_embedding_type == "FAIR":
            if data_set == "grc_proiel":
                w_code ="got"
            word_embedding_name = "cc."+w_code+".300.vec"
            if multilingual:
                word_embedding_name += ".proj"
        elif args.word_embedding_type == "CUSTOM":
            word_embedding_name = args.word_embedding_name

        word_path = os.path.join(WORD_EMBEDDING_PATH, word_embedding_name)
        #train_path = data_source_path+"/ud-treebanks-v2.2.renamed/ud-"+data_set+"/"+data_set+"-ud-train.conllu"
        #dev_path = data_source_path+"/ud-treebanks-v2.2.renamed/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
        #test_path = data_source_path+"/ud-treebanks-v2.2.renamed/ud-"+data_set+"/"+data_set+"-ud-test.conllu"
        train_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-train.conllu"
        dev_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-dev.conllu"
        test_path = data_source_path+"/ud-"+data_set+"/"+data_set+"-ud-test.conllu"


    # word embedding loading
    if random_init :
        word_dict, word_dim = None, 300
    else:
        word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    if agg_mode == "SUM" and lexicon is not None:
        assert pos_lexicon_dim == word_dim, "ERROR : pos_lexicon_dim should be same as word_dim"

    # parameters preprocessinng
    feats_space = None if not features else feats_space
    word_lenghts_return = True if char_mode=="RNN" else False
    lexicon_bool = False if lexicon is None else True

    conll_data = conllu_data
    create_dict = conllu_data.create_dict

    # create word, character, Part-of_Speech, Relation dictionaries
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_dict(alphabet_path,
                                                                            train_path,
                                                                            dev_path=dev_path,
                                                                            test_path=test_path,
                                                                            features=features,
                                                                            word_embed_dict=word_dict,
                                                                            lexicon=lexicon,
                                                                            lexicon_feats=lexicon_feats,
                                                                            lexicon_feats_inst2id=lexicon_feats_inst2id,
                                                                            dry_run=False, env=environment)
    #dictionary processing
    lexicon_alphabet, features_dictionary, pos_alphabet, num_feats_pred, num_lexicon_feats = preprocess_parameters(lexicon, features, lexicon_feats, pos_alphabet)

    if features and verbose:
        print("INFO  : len dictonary features :",  len(features_dictionary.instance2index))

    pred_writer_dev = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, features_dictionary)
    gold_writer_dev = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, features_dictionary)
    pred_writer_test = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, features_dictionary)
    gold_writer_test = CoNLLWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, features_dictionary)

    word_table = construct_word_embedding_table(word_dim=word_dim,
                                                word_alphabet=word_alphabet,
                                                unk_id=UNK_ID,
                                                freeze=False,
                                                word_dict=word_dict)
    # LEXICON POS SHOULD BE AS num_POS
    num_words, num_chars, num_pos, num_types = word_alphabet.size(), char_alphabet.size(),\
                                                  pos_alphabet.size(),  type_alphabet.size()

    model_full_path = os.path.join(model_path, "model")
    arguments = {"model_full_path":model_full_path, "data_set" : data_set, "word_embedding_name" : word_embedding_name, "lr" : lr,
                 "num_epoch" : num_epoch, "batch_size" : batch_size, "unk_replace" : unk_replace, "char_dim": char_dim,
                 "n_hidden_layers" : n_hidden_layers, "hidden_size":hidden_size, "attention_char":attention_char,
                 "features":features, "num_filters":num_filters, "pos_space" : pos_space,
                 "pos_lexicon_dim":pos_lexicon_dim, "clip": clip, "num_feats_pred": num_feats_pred,
                 "random_init": random_init, "lexicon": path_lexicon, "path_lexicon_feats": path_lexicon_feats,
                 "lexicon_feats": lexicon_feats, "feature_dim_lexicon": feature_dim_lexicon, "decay_rate": decay_rate,
                 "word_dim":word_dim, "char_mode":char_mode, "eps": eps,
                 "betas": betas, "gamma": gamma, "schedule": schedule, "max_decay": max_decay,
                 "feats_space":feats_space, "agg_mode": agg_mode,
                 "p_in":p_in, "p_out":p_out, "p_rnn":p_rnn, "p_class":p_class,
                 "lexicon_mode":lexicon_mode, "freeze_word_embedding": freeze_word_embedding}

    reporting_dir = os.path.join(REPORT_PATH,model_name)

    if not os.path.isdir(reporting_dir):
        os.mkdir(reporting_dir)
    
    if writing_logs and not RUN_LOCALLY:
        path_log = os.path.join(PROJECT_PATH, "models/log_run-"+run_id+"_rid-"+model_id+"_id"+".json")
        #path_log = "./models/log_run-"+run_id+"_rid-"+model_id+"_id"+".json"
        if os.path.isdir(path_log):
            with open(path_log,"r") as f:
                logs = json.load(f)        
        else:
            logs = {}
        logs[model_id] = {"model_full_path":model_full_path, "model_path":model_path, "arguments": arguments}
        with open(path_log, "w") as f:
            json.dump(logs, f)
        print("INFO LOGS : dumped at {} ".format(path_log))

    with open(os.path.join(reporting_dir, model_name + '-arguments.json'),"w") as f:
        json.dump(arguments, f)
    with open(model_path + "/" + model_name+'-arguments.json',"w") as f:
        json.dump(arguments, f)
    print("INFO MODEL arguments : ", arguments)
    print('INFO : writing arguments into {} and {} '.format(os.path.join(reporting_dir,model_name+"-arguments.json"),
                                                            model_path + "/" + model_name+'-arguments.json'))

    data_train = conll_data.read_data_to_variable(source_path=train_path, word_dictionary=word_alphabet,word_lenghts_return=word_lenghts_return,
                                                  char_dictionary=char_alphabet, pos_dictionary=pos_alphabet,
                                                  type_dictionary=type_alphabet, use_gpu=use_gpu,
                                                  symbolic_end=False,
                                                  symbolic_root=True, debug=debug, env=environment,
                                                  lexicon_dictionary=lexicon_alphabet,
                                                  feats_lex_dic=lexicon_feats_inst2id,
                                                  feats_dictionary=features_dictionary)

    data_dev = conll_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,type_alphabet,word_lenghts_return=word_lenghts_return,
                                                symbolic_end=False,lexicon_dictionary=lexicon_alphabet,
                                                feats_lex_dic=lexicon_feats_inst2id,
                                                feats_dictionary=features_dictionary,
                                                symbolic_root=True, env=environment,
                                                use_gpu=use_gpu)
    if score:
        data_test = conll_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet,
                                                     type_alphabet, use_gpu=use_gpu, env=environment,
                                                     word_lenghts_return=word_lenghts_return,
                                                     feats_dictionary=features_dictionary,
                                                     feats_lex_dic=lexicon_feats_inst2id,
                                                     lexicon_dictionary=lexicon_alphabet,
                                                     symbolic_root=True)
    # instanciate model
    model = NeuroTagger(word_dim=word_dim,char=True,
                        pos=False,
                        pos_space=pos_space,
                        embedd_word=word_table,
                        num_words=num_words,
                        char_dim=char_dim,
                        attention_char=attention_char,
                        feat_pred=features,num_feats_pred=num_feats_pred, feats_space=feats_space,
                        features_lexicon=lexicon_feats, feature_dim_lexicon=feature_dim_lexicon,
                        num_feats_lexicon=num_lexicon_feats,
                        num_chars=num_chars, 
                        num_pos=num_pos,
                        num_filters=num_filters,
                        kernel_size=3,lexicon=lexicon_bool,
                        pos_lexicon_dim=pos_lexicon_dim,
                        rnn_mode="LSTM",agg_mode=agg_mode,
                        hidden_size=hidden_size,
                        num_layers=n_hidden_layers,freeze_word_embedding=freeze_word_embedding,
                        p_in=p_in, p_out=p_out, p_rnn=p_rnn, char_mode=char_mode,
                        lexicon_mode=lexicon_mode,
                        p_class=p_class)
    if use_gpu:
        model.cuda()

    num_data = sum(data_train[1])
    num_batches = num_data // batch_size + 1
    params = model.parameters()

    if verbose:
        print("INFO Training {} num_batches per epoch".format(num_batches))

    params = filter(lambda param: param.requires_grad, params)
    optim = Adam(params, lr=lr, betas=(0.9, 0.9), weight_decay=0.0, eps=1e-8)

    start_time = time.time()
    decay_counter = 0.
    patient = 0

    n_total_inst_dev = 0.
    dev_pos_correct_mem = 0.

    error_dev = []
    error_test = []
    epochs_ls = []
    train_loss_ls = []
    error_train_ls = []
    dev_loss_ls = []
    test_loss_ls = []

    error_train_feat = []
    error_dev_feat = []
    error_test_feat = []

    time_very_start = time.time()

    if verbose:
        print("INFO TRAINING : starting training")

    # START TRAINING
    for ep in range(1, num_epoch+1):
        # writer
        pred_filename_dev_path = model_path+'/{}-{}_ep-pred_dev'.format(str(model_name), ep)
        gold_filename_dev_path = model_path+'/{}-{}_ep-gold_dev'.format(str(model_name), ep)
        pred_filename_test_path = model_path+'/{}-{}_ep-pred_test'.format(str(model_name), ep)
        gold_filename_test_path = model_path+'/{}-{}_ep-gold_test'.format(str(model_name), ep)

        pred_writer_test.start(pred_filename_test_path)
        gold_writer_test.start(gold_filename_test_path)
        gold_writer_dev.start(gold_filename_dev_path)
        pred_writer_dev.start(pred_filename_dev_path)

        model.train()
        if verbose:
            print("INFO TRAINING : start epoch {} ".format(ep))
        # START TRAINING BATCH
        train_err = 0
        train_count = 0
        train_pos_corr_total = 0
        train_pos_complete_match = 0
        n_total_inst_tr = 0
        train_feats_corr_total = 0
        train_total = 0
        train_feats_complete_match = 0
        for batch in range(1, num_batches+1):
            # default behavior
            # CHAR LENGHT NOT SUPPORTED
            chars_lenghts = None
            word, char, pos, _, _, masks, lengths, tags, feats, feats_input, \
            words_masks = conll_data.get_batch_variable(data_train, batch_size,
                                                        unk_replace=unk_replace,
                                                        lexicon=lexicon_bool,
                                                        word_mask_return=word_lenghts_return,
                                                        features=features,
                                                        features_input=lexicon_feats)
            if verbose_extreme:
                print("DATA : CHARS_LENGHS", chars_lenghts)
                print("DATA WORDS", word.size())
                print("DATA : CHARS", char.size())
                print("DATA words_masks", words_masks.size())
                print(" epoch {} batch {}".format(ep, batch))
                print(" --> length {} masks {}".format(lengths, masks))
                print(" word- {}  {} dim ".format(word, word.size()))
                print(" pos- {}  {} dim ".format(pos, pos.size()))

            optim.zero_grad()

            error = model.loss(input_word=word, input_char=char, pos=pos,
                               chars_lenghts=chars_lenghts, lexicon_tags=tags,
                               lexicon_feats=feats_input,
                               mask=masks, length=lengths,
                               hx=None, feats=feats,
                               words_masks=words_masks,
                               use_gpu=use_gpu,
                               debug=debug)
            error.backward()
            clip_grad_norm(model.parameters(), clip)
            optim.step()
            train_err += error.data[0]
            train_count += masks.sum().data[0]

            pos_pred_tr, feats_pred_tr = model.predict(input_word=word, input_char=char,lexicon_tags=tags,
                                                       feats_lexicon=feats_input,words_masks=words_masks,
                                                       mask=masks, length=lengths,use_gpu=use_gpu)
            # real data
            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            pos_pred_tr = pos_pred_tr.data.cpu().numpy()
            if features:
                feats = feats.data.cpu().numpy()
                feats_pred_tr = feats_pred_tr.data.cpu().numpy()

            (pos_corr_tr, total_tr, pos_complete_match_tr), \
            (feats_corr_tr, total_tr, feats_complete_match_tr),\
                                batch_size = evaluate(words=word, pos_pred=pos_pred_tr,
                                                      feats_pred=feats_pred_tr,feats=feats,
                                                      pos=pos, lengths=lengths)

            train_pos_corr_total += pos_corr_tr
            train_pos_complete_match += pos_complete_match_tr
            train_total += total_tr
            n_total_inst_tr += batch_size

            if features:
                train_feats_corr_total+=feats_corr_tr
                train_total+=total_tr
                train_feats_complete_match+=feats_complete_match_tr

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            if batch%50 == 0:
                print("INFO TRAINING :  train accuracy {}, train loss {},  batch {}, "
                      "epoch : {} , feats accuracy {} ".format(train_pos_corr_total*100/train_total, train_err/train_count,batch, ep,
                                                                                            train_feats_corr_total*100/train_total))
                print("INFO TIME {}s time_left ".format(time_left))

        if verbose:
            print("INFO : starting evaluation on dev, writing in {} for gold data and {} "
                  "for prediction ".format(gold_filename_dev_path, pred_filename_dev_path))
        model.eval()

        # START BATCH EVALUATION ON DEV
        dev_pos_corr_total = 0.
        dev_pos_complete_match = 0.
        dev_loss = 0.
        dev_count = 0.
        args = {"data": data_dev, "batch_size": batch_size, "lexicon": lexicon_bool,"features": features,
                "word_mask_return":word_lenghts_return,
                "features_input": lexicon_feats}

        dev_feats_corr_total = 0
        dev_total = 0+1e-8
        dev_feats_complete_match = 0

        for batch in conll_data.iterate_batch_variable(**args):

            word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks, lines_gold = batch

            heads_gold = heads_gold.data.cpu().numpy()
            types_gold = types_gold.data.cpu().numpy()
            if verbose_extreme and False:
                print("INFO evaluation data ")
                print("For {} batch_size o: input data is \n {} word - {} char -"
                      " {} heads - {} types - {} masks - {} lengths ".format(batch_size, word, char, "_", "_", masks, lengths))
            loss_dev = model.loss(input_word=word, input_char=char, pos=pos, feats=feats,
                                  chars_lenghts=chars_lenghts,lexicon_tags=tags,
                                  mask=masks, length=lengths, hx=None,lexicon_feats=feats_input,
                                  words_masks=words_masks,
                                  use_gpu=use_gpu,
                                  debug=debug)

            dev_loss += loss_dev.data[0]
            dev_count += masks.sum().data[0]

            pos_pred, feats_pred_dev = model.predict(input_word=word, input_char=char, lexicon_tags=tags,
                                                     feats_lexicon=feats_input, words_masks=words_masks,
                                                     mask=masks, length=lengths,use_gpu=use_gpu)
            # real data
            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            pos_pred = pos_pred.data.cpu().numpy()
            if features:
                feats_pred_dev = feats_pred_dev.data.cpu().numpy()
                feats = feats.data.cpu().numpy()
            (pos_corr, total, pos_complete_match),(feats_corr_dev, total_dev, feats_complete_match_dev), batch_size = evaluate(words=word, pos_pred=pos_pred,feats=feats,
                                                                                                                               feats_pred=feats_pred_dev,
                                                                                                                               pos=pos, lengths=lengths)
            dev_pos_corr_total += pos_corr
            dev_pos_complete_match += pos_complete_match
            dev_total += total
            n_total_inst_dev += batch_size

            if features:
                dev_feats_corr_total+=feats_corr_dev
                dev_total+=total_dev
                dev_feats_complete_match+=feats_complete_match_dev

            pred_writer_dev.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,
                                         raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)
            gold_writer_dev.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words, raw_lines=lines_gold,
                                         symbolic_root=True, symbolic_end=False)

        pred_writer_dev.write_buffer()
        gold_writer_dev.write_buffer()

        pred_writer_dev.close()
        gold_writer_dev.close()
        print("INFO : PERFORMANCE current on dev")
        print('--> Correct prediction {} compared to {} former ones, out of {} predictions/true value, '
              ' Correct % : {},  Mem Correct % {}   Correct Complete match %: {} , loss dev {} , {} accuracy feats '.format(
                                 dev_pos_corr_total, dev_pos_correct_mem, dev_total,
                                 dev_pos_corr_total*100 / dev_total,
                                 dev_pos_correct_mem*100/ dev_total,
                                 dev_pos_complete_match * 100 / n_total_inst_dev, dev_loss/dev_count, dev_feats_corr_total*100/dev_total))

        error_train_ls.append(train_pos_corr_total*100/train_total)
        train_loss_ls.append(train_err/train_count)
        error_dev.append(dev_pos_corr_total*100 / dev_total)
        dev_loss_ls.append(dev_loss/dev_count)
        epochs_ls.append(ep)

        error_train_feat.append(train_feats_corr_total*100/train_total)
        error_dev_feat.append(dev_feats_corr_total*100/dev_total)

        # START EVLAUATION ON TEST if model IMPROVED in accuracy


        former_accuracy = dev_pos_correct_mem  * 100 / dev_total

        if former_accuracy < dev_pos_corr_total * 100 / dev_total or not CONTROL_LEARNING :
            # model making progress --> we save
            dev_pos_correct_mem = dev_pos_corr_total
            dev_pos_correct_complete_match_mem = dev_pos_complete_match
            best_epoch = ep
            patient = 0
            # torch.save(network, model_name)
            print("INFO SAVING model at {} ".format(model_full_path))
            torch.save(model.state_dict(), model_full_path)

            if verbose:
                print( "INFO : starting  evaluation on test writing in {} for gold data and {} ,for prediction ".format(gold_filename_test_path, pred_filename_test_path))

            # START BATCH EVALUATION ON TEST if model IMPROVED in accuracy
            test_pos_corr_total = 0.
            test_pos_complete_match= 0.
            test_total = 1e-8
            n_total_inst_test = 0.
            test_loss = 0.
            test_count = 0
            test_feats_corr_total = 0
            test_feats_complete_match = 0

            args_test = {"data": data_test, "batch_size": batch_size, "lexicon": lexicon_bool,"features_input":lexicon_feats,"word_mask_return":word_lenghts_return,
                         "features": features}
            for batch in conll_data.iterate_batch_variable(**args_test):
                feats = None

                word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks,lines_gold = batch
                heads_gold = heads_gold.data.cpu().numpy()
                types_gold = types_gold.data.cpu().numpy()

                # report error
                loss_test = model.loss(input_word=word, input_char=char, pos=pos,feats=feats,
                                       chars_lenghts=chars_lenghts,lexicon_tags=tags,lexicon_feats=feats_input,
                                       mask=masks, length=lengths, hx=None,words_masks=words_masks,
                                       use_gpu=use_gpu,
                                       debug=debug)
                test_loss += loss_test.data[0]
                test_count += masks.sum().data[0]
                # prediction
                pos_pred, feats_pred_test = model.predict(input_word=word, input_char=char, lexicon_tags=tags,
                                                          feats_lexicon=feats_input, words_masks=words_masks,
                                                          use_gpu=use_gpu,
                                                          mask=masks, length=lengths)
                word = word.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                pos_pred = pos_pred.data.cpu().numpy()

                char = char.data.cpu().numpy()
                if features:
                    feats_pred_test = feats_pred_test.data.cpu().numpy()
                    feats = feats.data.cpu().numpy()

                # score
                (pos_corr, total, pos_complete_match), \
                (feats_corr_test, total, feats_complete_match_test), batch_size = evaluate(words=word, pos_pred=pos_pred,feats=feats,
                                                                                           feats_pred=feats_pred_test,
                                                                                           pos=pos, lengths=lengths)
                test_pos_corr_total += pos_corr
                test_pos_complete_match += pos_complete_match
                test_total += total
                n_total_inst_test += batch_size

                if features:
                    test_feats_corr_total+=feats_corr_test
                    test_feats_complete_match+=feats_complete_match_test

                pred_writer_test.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,
                                              raw_lines=lines_gold,
                                              symbolic_root=True, symbolic_end=False)
                gold_writer_test.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words,
                                              raw_lines=lines_gold,
                                              symbolic_root=True, symbolic_end=False)

            pred_writer_test.write_buffer()
            gold_writer_test.write_buffer()

            pred_writer_test.close()
            gold_writer_test.close()
            print("INFO : PERFORMANCE current on test ")

            print('--> Correct prediction {} out of {} predictions/true value,  Correct {} '
                  ', Correct Complete match {} , Test loss {} , {} accuracy feats'.format(
                   test_pos_corr_total, test_total,
                   test_pos_corr_total* 100 / test_total,
                   test_pos_complete_match * 100 / n_total_inst_test,test_loss/test_count,
                   test_feats_corr_total*100/test_total))

        elif CONTROL_LEARNING:
            # model performance degrading : not making progress or it's been a long time since we made progress
            # --> we load the former checkpoint
            if dev_pos_corr_total * 100 / dev_total < dev_pos_correct_mem * 100 / dev_total - 5 or patient >= schedule:
                # if the perf are degrading a lot or if it did not improve in a while or if we did not decay in a while
                # --> we load
                model.load_state_dict(torch.load(model_full_path))
                lr = lr * decay_rate
                optim = generate_optimizer(opt="Adam", lr=lr, params=model.parameters(),
                                           eps=eps, gamma=gamma, betas=betas)
                patient = 0
                decay_counter += 1
                if verbose:
                    print("INFO Training : Loading former checkpoint, decay_counter is {}".format(decay_counter))

                if decay_counter % double_schedule_decay == 0:
                    schedule *= 2
                    print('INFO TRAINING : schedule times 2  now {} '.format(schedule))
            else:
                # otherwise we wait
                patient += 1
                print("INFO Training : WAITING ")
        test_loss_ls.append(test_loss/test_count)
        error_test_feat.append(test_feats_corr_total*100/test_total)
        error_test.append(test_pos_corr_total* 100 / test_total)

        if reporting:
            plot_curves(error_train=error_train_ls, error_dev=error_dev,error_test=error_test,
                        loss_train=train_loss_ls,loss_dev=dev_loss_ls,
                        loss_test=test_loss_ls,
                        epochs=epochs_ls, epoch_max=ep,
                        model=model_name, data_set=data_set,
                        target_path=REPORT_PATH, save=True)
            plot_curves(error_train=error_train_feat, error_dev=error_dev_feat,error_test=error_test_feat,
                        info_suff="feats",
                        epochs=epochs_ls, epoch_max=ep,
                        model=model_name, data_set=data_set,
                        target_path=REPORT_PATH, save=True)
        if decay_counter == max_decay:
            print(" INFO : BREAKING training because decay {} reached max_decay ".format(decay_counter))

    print('------------------------------------------------------------------------------------------------------------')
    print('best dev pos {}, total {} , correct % {} , complete match {} , (epoch: {})'.format(
            dev_pos_correct_mem, dev_total, dev_pos_correct_mem * 100 / dev_total,
            dev_pos_correct_complete_match_mem * 100 / n_total_inst_dev,
            best_epoch))

    #
    gold_filename_test_path = model_path+'/{}-{}_ep-gold_test-FINAL'.format(str(model_name), ep)
    pred_filename_test_path = model_path+'/{}-{}_ep-pred_test-FINAL'.format(str(model_name), ep)
    pred_writer_test.start(pred_filename_test_path)
    gold_writer_test.start(gold_filename_test_path)
    # evaluate on test
    test_pos_corr_total = 0.
    test_pos_complete_match = 0.
    test_total = 0.
    n_total_inst_test = 0.

    if verbose:
        print("INFO : starting final evaluation on test writing in {} for gold data and {} ,"
              "for prediction ".format(gold_filename_test_path, pred_filename_test_path))
    # START FINAL EVALUATION
    for batch in conll_data.iterate_batch_variable(**args_test):

        word, char, pos, heads_gold, types_gold, masks, lengths, order_ids, raw_words, tags, feats, feats_input, words_masks, lines_gold = batch
        heads_gold = heads_gold.data.cpu().numpy()
        types_gold = types_gold.data.cpu().numpy()
        pos_pred, _ = model.predict(input_word=word,lexicon_tags=tags,words_masks=words_masks,
                                    input_char=char, feats_lexicon=feats_input,
                                    use_gpu=use_gpu,
                                    mask=masks, length=lengths)

        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        lengths = lengths.cpu().numpy()
        pos_pred = pos_pred.data.cpu().numpy()
        # score
        (pos_corr, total, pos_complete_match), _, batch_size = evaluate(words=word, pos_pred=pos_pred,
                                                                       pos=pos, lengths=lengths)

        test_pos_corr_total += pos_corr
        test_pos_complete_match += pos_complete_match
        test_total += total

        n_total_inst_test += batch_size
        pred_writer_test.store_buffer(word, pos_pred, heads_gold, types_gold, lengths, order_ids, raw_words,
                                      raw_lines=lines_gold,
                                      symbolic_root=True, symbolic_end=False)
        gold_writer_test.store_buffer(word, pos, heads_gold, types_gold, lengths, order_ids, raw_words,
                                      raw_lines=lines_gold,
                                      symbolic_root=True, symbolic_end=False)

    pred_writer_test.write_buffer()
    gold_writer_test.write_buffer()
    pred_writer_test.close()
    gold_writer_test.close()
    print("INFO : PERFORMANCE final on test ")
    print('--> Correct prediction {} out of {} predictions/true value,  Correct {} '
          ', Correct Complete match {} % '.format(
           test_pos_corr_total, test_total,
           test_pos_corr_total* 100 / test_total,
           test_pos_complete_match * 100 / n_total_inst_test))
    time_very_end = -(time_very_start-time.time())/60
    print("TIME : time_very_end {} ".format(time_very_end))

    if writing_logs and not RUN_LOCALLY:
        if os.path.isdir(path_log):
            with open(path_log,"r") as f:
                logs = json.load(f)        
        logs[model_id]["time_training"] = time_very_end
        logs[model_id]["arguments"]["lr_last"] = lr
        logs[model_id]["perf_gold_test"] = test_pos_corr_total* 100 / test_total
        logs[model_id]["perf_gold_dev"] = dev_pos_corr_total*100 / dev_total
        with open(path_log, "w") as f:
            json.dump(logs, f)
        print("INFO LOGS : dumped at {} ".format(path_log))

if __name__ == "__main__":
    main()





