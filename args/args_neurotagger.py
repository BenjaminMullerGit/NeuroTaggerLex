
import argparse

DEFAULT_PARAMETERS_COMMON={"run_id": "RUN_TEST"}

def sanity_check_args(args):

    assert args.word_embedding_type in ["CUSTOM", "FAIR"], "word_embedding_type should be in ['CUSTOM', 'FAIR']"

    if args.word_embedding_type == "CUSTOM":
        assert args.word_embedding_name is not None, "ERROR : in custom MODE you have to provide " \
                                                     "the --word_embedding_name file name"
    if int(args.use_lexicon) or int(args.lexicon_feats):
        assert args.lexicons_path is not None, "ERROR : if use_lexicon or lexicons_feats : --lexicons_path required "
    else:
        assert args.lexicons_path is None, "ERROR : --lexicons_path should not specify as use_lexicon or lexicon_feats is set to true"

def sanity_check_score_args(args):
    if args.multilingual is not None:
        if args.multilingual == "1":
            assert args.lexicon_path is not None, "ERROR : --lexicon_path should be provided in multilingual mode"
        else:
            assert args.lexicon_path is None, " ERROR : lexicon_path should not be specify"
    else:
        assert args.lexicon_path is None, " ERROR : lexicon_path should not be specify"

def fit_args(required):

    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')

    args_parser.add_argument('--data_set', help='data set with normalized naming', required=required)
    args_parser.add_argument('--model_id', help='model id ', required=required)
    args_parser.add_argument('--run_id', help='run id ', required=False, default= DEFAULT_PARAMETERS_COMMON["run_id"])
    args_parser.add_argument('--data_source_path', help='run id ', required=required)
    args_parser.add_argument('--word_embedding_name', help='run id ', required=False)
    args_parser.add_argument('--word_embedding_type', help='run id ', required=False, default="FAIR")
    args_parser.add_argument('--lexicons_path', help='run id ', required=False)
    args_parser.add_argument('--use_lexicon', help='run id ', required=False, default=0)
    args_parser.add_argument('--lexicon_feats', help='run id ', required=False,default=0)
    args_parser.add_argument('--prerun', help='run id ', required=False, default=0)
    args_parser.add_argument('--hidden_size', help='hidden_size of word level recurrent cell', required=False, default=200)
    args_parser.add_argument('--attention_char', help='performing character level attention', required=False, default=0)
    args_parser.add_argument('--char_mode', help='cell to process character information', required=False, default="RNN")
    args_parser.add_argument('--num_epoch', help='number of epochs', required=False, default=300)
    args_parser.add_argument('--batch_size', help='batch dimension', required=False, default=6)
    args_parser.add_argument('--lexicon_mode', help=' within [n-got, continuous] ', required=False, default="continuous")
    args_parser.add_argument('--agg_mode', help=' within [SUM, CAT] ', required=False, default="SUM")

    args = args_parser.parse_args()

    sanity_check_args(args)

    return args

def score_args():
    args_parser = argparse.ArgumentParser(description='')
    args_parser.add_argument('--data_set', help='data set with normalized naming', required=True)
    args_parser.add_argument('--model_id', help='model id ', required=True)
    args_parser.add_argument('--run_id', help='run id ', required=False, default=DEFAULT_PARAMETERS_COMMON["run_id"])
    args_parser.add_argument('--data_source_path_train', help=' ', required=True)
    args_parser.add_argument('--data_source_path_test', help=' ', required=True)
    args_parser.add_argument('--test_file_name_custom', help=' ', required=False)
    args_parser.add_argument('--multilingual', help=' if want to do transfer from one language to another ', required=False)
    args_parser.add_argument('--data_set_model', help=' ', required=False)
    args_parser.add_argument('--lexicon_path', help='in multilingual model : needed '
                                                    'to provide the target language lexicon ', required=False)
    args = args_parser.parse_args()
    sanity_check_score_args(args)

    return args