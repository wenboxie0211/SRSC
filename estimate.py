from scipy.special import comb
def rand_index(label, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for m in range(len(label)):
        # for n in range(0, m):
        #     if label[m] == label[n]:
        #         if pred[n] == pred[m]:
        #             TP += 1
        #         # else:
        #         #     FN += 1
        #     else:
        #         if pred[n] != pred[m]:
        #             TN += 1
        #         # else:
        #         #     FP += 1
        for n in range(m):
            if label[m] == label[n] and pred[n] == pred[m]:
                TP += 1
                # else:
                #     FN += 1
            elif label[m] != label[n] and pred[n] != pred[m]:
                TN += 1
                # else:
                #     FP += 1
    return (TP + TN) / comb(len(label), 2)
