



def evaluate(words, pos_pred, pos, lengths, feats=None,feats_pred=None):

    batch_size, _ = words.shape
    pos_corr = 0.
    pos_complete_match = 0.
    start, end= 0, 0
    total = 0.

    feats_corr = 0.
    feat_complete_match=0

    for i in range(batch_size):
        cm = 1.
        cm_feat = 1
        for j in range(start, lengths[i] - end):
            total += 1
            if feats is not None and feats_pred is not None:
                if feats[i, j] == feats_pred[i, j]:
                    feats_corr += 1
                else:
                    cm_feat = 0
            if pos[i, j] == pos_pred[i, j]:
                pos_corr += 1
            else:
                cm = 0
        pos_complete_match += cm
        feat_complete_match+=cm_feat

    return (pos_corr, total, pos_complete_match),(feats_corr,total,feat_complete_match), \
            batch_size
