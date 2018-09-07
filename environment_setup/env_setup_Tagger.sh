#!/usr/bin/env bash


export PROJECT_PATH="./"
export REPORT_PATH=""
export WORD_EMBEDDING_PATH=""

conda create -n tagger python=3.6 anaconda
source activate tagger
conda install pytorch=0.3.1 -c soumith
conda install gensim=3.4
conda install tqdm
conda install -c conda-forge matplotlib 

wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64
chmod +x ./jq
cp jq /usr/bin



