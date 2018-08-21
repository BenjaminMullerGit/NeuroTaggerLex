#!/usr/bin/env bash


# RUNNING AND EVALUATING A SINGLE MODEL 

export PROJECT_PATH=/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/NeuroTagger
export REPORT_PATH=/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/NeuroTagger/reports
export WORD_EMBEDDING_PATH=/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/data/ud/supdata/ud-2.0-baselinemodel-train-embeddings

source activate neuroNLP2_3

MODEL_ID=model
RUN_ID=run
DATA_SET=da_ddt_sample

python $PROJECT_PATH/fit.py --data_set ${DATA_SET} --run_id $RUN_ID --model_id ${MODEL_ID}  --data_source_path ../data/release-2.2-st-train-dev-data-NORMALIZED --word_embedding_type CUSTOM --prerun 1 --word_embedding_name en.skip.forms.50.vectors --lexicon_feats 0 --use_lexicon 0
python $PROJECT_PATH/score.py --data_set ${DATA_SET} --run_id $RUN_ID --model_id ${MODEL_ID} --data_source_path ../data/release-2.2-st-train-dev-data-NORMALIZED --test_file_name_custom ud-${DATA_SET}/${DATA_SET}-ud-dev.conllu


python $PROJECT_PATH/evaluation/conll18_ud_eval.py $PROJECT_PATH/models/${RUN_ID}_rid-${MODEL_ID}_id-${DATA_SET}_data/${MODEL_ID}-SCORE-pred_test ../data/release-2.2-st-train-dev-data-NORMALIZED/ud-${DATA_SET}/${DATA_SET}-ud-dev.conllu -v


# RUNNING AND EVALUATING SEVERAL MODELS 

sh $PROJECT_PATH/metal_all.sh 
