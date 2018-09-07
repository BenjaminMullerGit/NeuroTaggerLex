
# NeuroTaggerLex : Neural Tagger with Lexicon

## Description 

### Model 

This repository includes a Part-of-Speech and morphological features model called NeuroTaggerLex. It was used by the ParisNLP18 team for the CoNLL 2018 Shared Task (http://universaldependencies.org/conll18)

In short, the NeuroTaggerLex model is a deep learning sequence prediction model. It consists in a first word embedding module which learns a representation of character information, word information , and external lexical information.  This representation is then fed to a Bi-LSTM layer followed by a classification layer with two heads, one for Univeral Part-of-Speech tag and one for a restricted list of morphological features.

This model supports the Universal Dependency format (http://universaldependencies.org/guidelines.html)

#### Repository 

This repository allows code for training and evaluating the NeuroTaggerLex model. 


## Environment set up  

To set up the environment, first define in ./environment_setup/env_setup_Tagger.sh  the paths PROJECT_PATH where your project is located, REPORT_PATH where the reports will be located and WORD_EMBEDDING_PATH where the word embedding will be downloaded

then run : 

`sh ./environment_setup/env_setup_Tagger.sh 

## Data, word embedding and lexicons

### CONLL-U 

`cd ./data `

`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz`

`tar -xvzf ud-treebanks-v2.2.tgz`

<!---
.. renaming all the folders 
-->

### Word Embedding 



`cd WORD_EMBEDDING_PATH`

with lang="fr" for instance

`wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.$lang.300.vec.gz`

`gunzip cc.$lang.300.vec.gz`

### Lexicons 

`cd ./lexicons `

`wget http://atoll.inria.fr/~sagot/UDLexicons.0.2.zip`

`unzip UDLexicons.0.2.zip`

<!---
..preprocessing commands
-->

## Train and evaluate a model 

Follow ./demo.sh 

# Others 

## Resources
* [Raw Treebanks](http://universaldependencies.org/conll18/data.html)
* [UDLexicons](http://pauillac.inria.fr/~sagot/index.html#udlexicons)
* [FastText Word Embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

## Acknowledgements
* [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) from XuezheMax
* [Chu-Liu-Edmonds](https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/mst/mst.ipynb) from bastings

## License
ELMoLex is GPL-licensed.


