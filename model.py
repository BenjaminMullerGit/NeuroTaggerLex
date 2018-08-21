import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_tools.layers import VarMaskedLSTM
from nn_tools.sparse import Embedding
from i_o.conllu_data import n_hot_tf
from torch.autograd import Variable



class NeuroTagger(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_pos, num_filters, 
                 pos_space,
                 #pos_dim, num_labels,
                 kernel_size, rnn_mode, hidden_size, num_layers,
                 pos_lexicon_dim=None,
                 features_lexicon=False, num_feats_lexicon=None, feature_dim_lexicon=0,
                 feats_space = None, num_feats_pred=0,feat_pred=False,
                 lexicon=False,freeze_word_embedding=False,agg_mode="CAT",
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33,
                 p_rnn=(0.33, 0.33), p_class=0.3, char_mode="CNN",
                 attention_char = False,
                 lexicon_mode="continuous",
                 pos=False, char=True):
        """
        Definition method : we only define the blocks of the network : each layer (drop-out), 
        and their dimensions
        """
        
        super(NeuroTagger, self).__init__()

        assert lexicon_mode in ["continuous", "n-hot"]
        assert char_mode in ["CNN", "RNN"]
        assert agg_mode in ["CAT", "SUM"]
        assert rnn_mode == "LSTM", "ERROR : only LSTM supported "
        if attention_char:
            assert char_mode == "RNN", "ERROR : attention is related to recurrent cell"
        if lexicon_mode == "n-hot":
            assert not features_lexicon and lexicon, "ERROR : n-hot only supported on lexicons tags"
            assert agg_mode == "CAT", " n-hot is only supported with agg_mode as CAT "
        if lexicon:
            assert pos_lexicon_dim is not None
        else:
            pos_lexicon_dim = 0
        if features_lexicon:
            assert num_feats_lexicon is not None \
                   and feature_dim_lexicon > 0, "INFO about feats INPUT space are required to use them"
        else:
            if feature_dim_lexicon > 0:
                print("WARNING : wrong parameter passed, setting features_dim_lexicon to 0 ")
            feature_dim_lexicon = 0

        if feat_pred:
            assert feats_space is not None and num_feats_pred > 0, "INFO about feats OUTPUT space are required"
        # exact same way
        self.lexicon = lexicon
        self.features_lexicon = features_lexicon
        self.lexicon_mode = lexicon_mode
        self.feat_pred = feat_pred
        self.agg_mode = agg_mode
        self.attention_char = attention_char
        self.char_mode = char_mode
        self.feat_pred = feat_pred
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)

        #self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos) if pos else None
        if lexicon and lexicon_mode == "continuous":
            self.lexicon_embed = Embedding(num_pos, pos_lexicon_dim, init_embedding=None)
        elif lexicon and lexicon_mode == "n-hot":
            self.num_pos = num_pos
            print("WARNING : pos_lexicon_dim set to {} num_pos in n-hot mode".format(num_pos))
            pos_lexicon_dim  = num_pos
        else:
            self.lexicon_embed = None

        if freeze_word_embedding:
            for param in self.word_embedd.parameters():
                param.requires_grad = False

        self.feature_embed = Embedding(num_feats_lexicon, feature_dim_lexicon, init_embedding=None) if features_lexicon else None
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char) if char else None

        if char_mode == "CNN":
            self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
        elif char_mode == "RNN":
            RNN_char = VarMaskedLSTM
            dim_multi = 1
            self.rnn_char = RNN_char(input_size=char_dim,hidden_size=num_filters,
                                     num_layers=1,bias=True,
                                     batch_first=True, bidirectional=False,
                                     dropout=(0., 0.3))

            if self.attention_char:
                # attention implies concatanating cell and attention vector
                dim_multi = 2
            self.char_proj = nn.Linear(num_filters*dim_multi, word_dim)
            if self.attention_char:
                self.soft_max = nn.Softmax(dim=1)
                self.attention_inner = nn.Linear(bias=False, in_features=num_filters, out_features=1)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.pos = pos
        self.char = char

        RNN = VarMaskedLSTM
        dim_enc = word_dim

        if char:
            if char_mode == "CNN" and agg_mode=="CAT":
                dim_enc += num_filters
                dim_enc += pos_lexicon_dim
                dim_enc += feature_dim_lexicon
            elif char_mode == "CNN" and agg_mode=="SUM":
                dim_enc = word_dim
                assert num_filters == word_dim
                if features_lexicon:
                    assert num_filters == feature_dim_lexicon
                # same for lexicon dim
            elif char_mode == "RNN" and agg_mode=="CAT":
                dim_enc += word_dim # because of the projection
                dim_enc += feature_dim_lexicon
                dim_enc += pos_lexicon_dim
            elif char_mode == "RNN" and agg_mode == "SUM":
                dim_enc = word_dim
                if features_lexicon:
                    assert word_dim == feature_dim_lexicon
            else:
                raise ValueError("char_mode agg_mode not consistent")

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, 
                       batch_first=True,
                       bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        # MLP
        self.mlp_pos_lin_NOUN = nn.Linear(out_dim, pos_space)
        # relu added in the flow
        self.dropout_output_layer = nn.Dropout(p=p_class)
        self.output_pos = nn.Linear(pos_space, num_pos)

        if self.feat_pred:
            self.mlp_pos_lin_FEATS = nn.Linear(out_dim, feats_space)
            # relu added in the flow
            self.output_feats = nn.Linear(feats_space, num_feats_pred)

    def forward(self,input_word, input_char, input_pos=None, mask=None,
                chars_lenghts=None, feats_lexicon=None,
                lexicon_tags=None, words_masks=None,
                length=None, hx=None,use_gpu=False):
        """
        we define here the data flow
        NB : the activations function are defined here 
        """
        if self.char_mode == "RNN":
            assert words_masks is not None

        word = self.word_embedd(input_word)
        # apply dropout on input
        word = self.dropout_in(word)
        input = word

        if self.features_lexicon:
            # embedding lexicon morphological feature tags
            feats_lexicon = self.feature_embed(feats_lexicon)
            # computing the mean
            feats_lexicon = feats_lexicon.mean(dim=2)
        if self.lexicon:
            # embedding pos lexicon tags
            if self.lexicon_mode == "continuous":
                lexicons = self.lexicon_embed(lexicon_tags)
                # computing the mean
                lexicons = lexicons.mean(dim=2)
            elif self.lexicon_mode == "n-hot":
                lexicons = n_hot_tf(lexicon_tags, n_label=self.num_pos, use_gpu=use_gpu)
        if self.char:
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            #print("CHAR SIZE", char_size)
            if self.char_mode == "CNN":
                char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
                # put into cnn [batch*length, char_filters, char_length]
                # then put into maxpooling [batch * length, char_filters]
                char, _ = self.conv1d(char).max(dim=2)
                # reshape to [batch, length, char_filters]
                char = torch.tanh(char).view(char_size[0], char_size[1], -1)
                # apply dropout on input
                char = self.dropout_in(char)
            if self.char_mode == "RNN":
                # first transform to [batch *length, char_length, char_dim]
                char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3])
                words_masks_size = words_masks.size()
                words_masks = words_masks.view(words_masks_size[0] * words_masks_size[1], words_masks_size[2])
                hn_full, (last_hn, last_cell) = self.rnn_char(char, words_masks)
                hn_full_att = hn_full.contiguous()
                hn_full = hn_full[:, char_size[2]-1, :]
                hn_full = hn_full.contiguous()
                hn_full = hn_full.view(char_size[0], char_size[1], -1)
                #print("HN FULL", hn_full)
                if self.attention_char:
                    hn_full_size = hn_full_att.size()
                    #print("HIDDEN FULL ", hn_full_att.size())
                    #print("HIDDN", hn_full_att[-1,-1,:])
                    # TODO : confirm this transformation
                    hidden = self.attention_inner(hn_full_att)
                    # TODO : confirm that you don't need masking on attention weights as hidden states have been masked already (they are masked but how correectly?)
                    #print("HIDDEN INNER", hidden.size())
                    hn_full_att = hn_full_att.view(hn_full_size[0], hn_full_size[2], hn_full_size[1])
                    #print("HIDDEN FULL VIEWED ", hn_full_att.size())
                    soft = self.soft_max(hidden)
                    #print("SOFT", soft.size())
                    # checking if mask works
                    #print("SOFT", soft[:50, 10, 0])
                    #print("MASK", words_masks.size())
                    #print("EXPAND", words_masks.expand_as(hn_full_att))
                    #words_masks = words_masks.view(words_masks.size(0), words_masks.size(1), 1)
                    #print('SUM BEFORE MASK', torch.sum(soft,dim=1))
                    #soft = torch.mul(soft, words_masks)
                    #print("SOFT MASKED", soft[:50, 10, 0])
                    #print('SUM', torch.sum(soft,dim=1))
                    new_hidden = torch.bmm(hn_full_att, soft)
                    #print("new_hidden", new_hidden.size())
                    size = new_hidden.size()
                    new_hidden = new_hidden.view(char_size[0], char_size[1], size[1])
                    #print("attention hidden ", new_hidden.size())
                    #print("hn_full last hidden", hn_full.size())
                    last_cell_size = last_cell.size()
                    #print("last cell size", last_cell_size)
                    last_cell = last_cell.view(char_size[0], char_size[1], last_cell_size[2]*last_cell_size[0])
                    #print("last cell", last_cell.size())
                    char = torch.cat([last_cell, new_hidden], dim=2)
                    #print("LAST CELL", last_cell)
                else:
                    char = hn_full
                char = self.char_proj(char)

            if self.agg_mode == "SUM":
                input = input+char
                if self.lexicon:
                    input = input+lexicons
                if self.features_lexicon:
                    input = input+feats_lexicon
            elif self.agg_mode == "CAT":

                if self.lexicon and not self.features_lexicon:
                    if self.lexicon_mode == "n-hot":
                        lexicons = Variable(lexicons)
                    input = torch.cat([input, char, lexicons], dim=2)
                elif not self.lexicon and self.features_lexicon:
                    input = torch.cat([input, char, feats_lexicon], dim=2)
                elif self.lexicon and self.features_lexicon:
                    input = torch.cat([input, char, lexicons, feats_lexicon], dim=2)
                else:
                    input = torch.cat([input, char], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask, hx=hx)
        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        # output size [batch, length, arc_space]
        pos_h = F.elu(self.mlp_pos_lin_NOUN(output))
        pos_h = self.dropout_output_layer(pos_h)
        score_POS = self.output_pos(pos_h)

        if self.feat_pred:
            feat_h = F.elu(self.mlp_pos_lin_FEATS(output))
            # dropout_output_layer renamed dropout_class
            feat_h = self.dropout_output_layer(feat_h)
            score_FEAT = self.output_feats(feat_h)
        else:
            score_FEAT = None

        return score_POS, score_FEAT

    def predict(self, input_word, input_char, mask=None, length=None, hx=None, chars_lenghts=None, lexicon_tags=None, feats_lexicon = None, words_masks=None,use_gpu=False):
        score_POS, score_FEAT = self.forward(input_word=input_word, input_char=input_char, mask=mask, length=length, hx=hx,
                                             lexicon_tags=lexicon_tags, chars_lenghts=chars_lenghts,feats_lexicon=feats_lexicon,
                                             words_masks=words_masks,use_gpu=use_gpu)
        _, indices = torch.max(score_POS, 2)
        if self.feat_pred:
            _, indices_feat = torch.max(score_FEAT, 2)
        else:
            indices_feat = None
        return indices, indices_feat

    def loss(self,input_word, input_char, pos, mask=None, feats=None, length=None, hx=None, debug=False,
             use_gpu=False, lexicon_tags=None, lexicon_feats=None,
             chars_lenghts=None, words_masks=None):
        
        score_POS, score_Feats = self.forward(input_word, input_char, mask=mask, length=None, hx=None,
                                              chars_lenghts=chars_lenghts,
                                              lexicon_tags=lexicon_tags, feats_lexicon=lexicon_feats,
                                              words_masks=words_masks,use_gpu=use_gpu)

        _, indices = torch.max(score_POS, 2)

        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            minus_mask = minus_mask.unsqueeze(-1)
            minus_mask_pos = minus_mask.expand(score_POS.size()[0], score_POS.size()[1], score_POS.size()[2])
            if self.feat_pred:
                minus_mask_feat = minus_mask.expand(score_Feats.size()[0], score_Feats.size()[1], score_Feats.size()[2])
                score_Feats = score_Feats + minus_mask_feat
            score_POS = score_POS + minus_mask_pos
        log_soft = F.log_softmax(score_POS, dim=2)
        if self.feat_pred:
            assert feats is not None, 'ERROR feats labels must be provided to learn to predict them '
            log_soft_feats = F.log_softmax(score_Feats, dim=2)

        log_soft = log_soft.view(score_POS.size()[0]*score_POS.size()[1], score_POS.size()[2])
        pos = pos.view(score_POS.size()[0]*score_POS.size()[1])
        if self.feat_pred:
            log_soft_feats = log_soft_feats.view(score_Feats.size()[0]*score_Feats.size()[1], score_Feats.size()[2])
            feats = feats.view(score_Feats.size()[0]*score_Feats.size()[1])
        if use_gpu:
            indexes = torch.arange(score_POS.size(0)*score_POS.size(1)).cuda().long()
            if self.feat_pred:
                indexes_feat = torch.arange(score_Feats.size(0)*score_Feats.size(1)).cuda().long()
        else:
            indexes = torch.arange(score_POS.size(0)*score_POS.size(1)).long()
            if self.feat_pred:
                indexes_feat = torch.arange(score_Feats.size(0)*score_Feats.size(1)).long()

        log_soft = log_soft[indexes, pos]
        log_soft = log_soft.view(score_POS.size(0),score_POS.size(1))
        if self.feat_pred:
            log_soft_feats = log_soft_feats[indexes_feat, feats]
            log_soft_feats = log_soft_feats.view(score_Feats.size(0),score_Feats.size(1))
        if mask is not None :
            log_soft = log_soft*mask
            if self.feat_pred :
                log_soft_feats = log_soft_feats*mask
        num = mask.sum()
        loss = -log_soft.sum()/num
        if self.feat_pred:
            loss += -log_soft_feats.sum()/num

        return loss
