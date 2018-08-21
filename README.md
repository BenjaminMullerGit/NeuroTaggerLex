
# NeuroTagger with lexicon

## Environment set up  

sh environment_setup/env_setup_tagger.sh 

## Data, word embedding and lexicons

### CONLL-U 
cd data 
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz
tar -xvzf ud-treebanks-v2.2.tgz

.. renaming all the folders 

### Word Embedding 

cd WORD_EMBEDDING_PATH

lang="fr"
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.$lang.300.vec.gz
gunzip cc.$lang.300.vec.gz

### Lexicons 

cd lexicons 
wget http://atoll.inria.fr/~sagot/UDLexicons.0.2.zip
unzip UDLexicons.0.2.zip
..preprocessing commands

## Train and evaluate a model 

Define the three environment variables  PROJECT_PATH, REPORT_PATH and  WORD_EMBEDDING_PATH
Then you can follow demo.sh by running fit.py score.py 

