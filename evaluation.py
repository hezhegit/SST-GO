import obonet
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score as aupr
import utils
from matplotlib import pyplot as plt

plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', family='arial')


go_graph = obonet.read_obo(open("/home/415/hz_project/SSGO/data/go-basic.obo", 'r'))



def propagate_binary_to_ancestors(Y_binary, goterms, go_graph):

    N_prot, N_term = Y_binary.shape
    Y_expanded = np.zeros_like(Y_binary, dtype=np.int32)

    term2idx = {go: idx for idx, go in enumerate(goterms)}
    ancestors_idxs = []
    for j, goterm in enumerate(goterms):
        anc = nx.ancestors(go_graph, goterm)
        all_anc = set(anc)
        all_anc.add(goterm)
        idx_list = [term2idx[g] for g in all_anc if g in term2idx]
        ancestors_idxs.append(idx_list)

    for i in range(N_prot):
        direct_js = np.where(Y_binary[i] == 1)[0]
        for j in direct_js:
            for k in ancestors_idxs[j]:
                Y_expanded[i, k] = 1

    return Y_expanded


def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    # num_ru[i] = sum_{g ∈ T_i \ P_i} IC(g)
    num_ru = np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    # denom[i] = sum_{g ∈ T_i ∪ P_i} IC(g)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)

    # 对 denom=0 的蛋白，将 denom_corr=1，之后再把 nru[i]=0
    denom_corr = np.where(denom == 0, 1.0, denom)
    nru = num_ru / denom_corr
    nru[denom == 0] = 0.0

    if avg:
        valid_mask = (denom > 0)
        if np.any(valid_mask):
            return np.mean(nru[valid_mask])
        else:
            return 0.0
    return nru


def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    """
    计算 MI 向量或标量：
      mi_i = [sum_{g ∈ P_i \ T_i} IC(g)] / [sum_{g ∈ T_i ∪ P_i} IC(g)]

    如果 avg=True，则返回 np.mean(mi_i)(仅对 denom>0 的 i 取平均)，否则返回 mi_i 向量。
    denom=0 时先置为 1 来避免除零，然后将对应 nmi[i]=0、不参与 avg。
    """
    num_mi = np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)

    denom_corr = np.where(denom == 0, 1.0, denom)
    nmi = num_mi / denom_corr
    nmi[denom == 0] = 0.0

    if avg:
        valid_mask = (denom > 0)
        if np.any(valid_mask):
            return np.mean(nmi[valid_mask])
        else:
            return 0.0
    return nmi


def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi=False):
    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)

    if avg:
        ru_avg = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, True)
        mi_avg = normalizedMisInformation(Ytrue, Ypred, termIC, True)
        sd_avg = np.sqrt(ru_avg ** 2 + mi_avg ** 2)
        if returnRuMi:
            return [ru_avg, mi_avg, sd_avg]
        else:
            return sd_avg
    else:
        if returnRuMi:
            return [ru, mi, sd]
        else:
            return sd


def _cafa_go_aupr(labels, preds, task):
    N_prot, _ = labels.shape
    _, goterms_all, _, _ = utils.load_GO_annot("/home/415/hz_project/SSGO/data/nrPDB-GO_2019.06.18_annot.tsv")
    goterms = np.asarray(goterms_all[task])

    labels_expanded = propagate_binary_to_ancestors(labels, goterms, go_graph)
    prot2goterms = {}
    for i in range(N_prot):
        idxs = np.where(labels_expanded[i] == 1)[0]
        prot2goterms[i] = set(goterms[idxs])

    F_list = []
    AvgPr_list = []
    AvgRc_list = []
    thresh_list = []

    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        predictions = (preds > t).astype(int)
        preds_expanded = propagate_binary_to_ancestors(predictions, goterms, go_graph)

        m = 0
        precision = 0.0
        recall = 0.0

        for i in range(N_prot):
            pred_gos = set()
            for j in np.where(preds_expanded[i] == 1)[0]:
                pred_gos.add(goterms[j])
            true_gos = prot2goterms[i]
            num_pred = len(pred_gos)
            num_true = len(true_gos)
            if num_pred > 0 and num_true > 0:
                overlap = len(pred_gos.intersection(true_gos))
                if overlap > 0:
                    m += 1
                    precision += float(overlap) / num_pred
                    recall += float(overlap) / num_true

        if m > 0:
            AvgPr = precision / m
            AvgRc = recall / N_prot
            if (AvgPr + AvgRc) > 0:
                F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                F_list.append(F_score)
                AvgPr_list.append(AvgPr)
                AvgRc_list.append(AvgRc)
                thresh_list.append(t)

    return (
        np.asarray(AvgRc_list),
        np.asarray(AvgPr_list),
        np.asarray(F_list),
        np.asarray(thresh_list)
    )


def _function_centric_aupr(Y_true, Y_pred, task):
    _, goterms_all, _, _ = utils.load_GO_annot("/home/415/hz_project/SSGO/data/nrPDB-GO_2019.06.18_annot.tsv")
    goterms = np.asarray(goterms_all[task])

    keep_goidx = np.where(Y_true.sum(axis=0) > 0)[0]
    Y_true_f = Y_true[:, keep_goidx]
    Y_pred_f = Y_pred[:, keep_goidx]

    return aupr(Y_true_f, Y_pred_f, average='macro')


def _protein_centric_fmax(Y_true, Y_pred, task):
    Recall, Precision, Fscore, thresholds = _cafa_go_aupr(Y_true, Y_pred, task)
    return Fscore, Recall, Precision, thresholds


def fmax(Y_true, Y_pred, task):
    Fscore, _, _, _ = _protein_centric_fmax(Y_true, Y_pred, task)
    return np.max(Fscore) if Fscore.size > 0 else 0.0


def macro_aupr(Y_true, Y_pred, task):
    return _function_centric_aupr(Y_true, Y_pred, task)


def smin(termIC, Y_true, Y_pred, task):
    thresholds = np.linspace(0.01, 0.99, 99)
    ss = np.zeros(len(thresholds), dtype=np.float32)

    for i, t in enumerate(thresholds):
        preds_binary = (Y_pred >= t).astype(int)
        sd_t = normalizedSemanticDistance(
            Y_true,         
            preds_binary,   
            termIC,
            avg=True,
            returnRuMi=False
        )
        ss[i] = sd_t

    return np.min(ss)

