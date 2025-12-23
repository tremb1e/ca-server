import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,f1_score
import pickle
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def utils_eer(y_true, y_pred, return_threshold=False):
    """Calculate the Equal Error Rate.

    Based on https://stackoverflow.com/a/49555212, https://yangcha.github.io/EER-ROC/
    and https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object

    Arguments:
        y_true {np.array}  -- Actual labels
        y_pred {np.array}  -- Predicted labels or probability

    Returns:
        float              -- Equal Error Rate
    """
    #print("the y_pred is:", y_pred)
    #print("the y_test is:", y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    index = 0
    for i in range (len(tpr)):
        if tpr[i] > 0.970:
            index = i
            break

    #print("############################################# thresholds:", thresholds[index])
    y = np.greater_equal(y_pred, thresholds[index])

    gt_outlier = np.logical_not(y_true)

    true_positive = np.sum(np.logical_and(y, y_true))
    true_negative = np.sum(np.logical_and(np.logical_not(y), gt_outlier))
    false_positive = np.sum(np.logical_and(y, gt_outlier))
    false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
    total_count = true_positive + true_negative + false_positive + false_negative
    #print("the true_positive, the true_negative, the false_positive, the false_negative is:", true_positive, true_negative, false_positive, false_negative)
    accuracy = 100 * (true_positive + true_negative) / total_count
    far = 100 * false_positive/(true_negative+false_positive)
    frr = 100 * false_negative/(false_negative + true_positive)
    f1 = get_f1(true_positive, false_positive, false_negative)
    #print("#################the far, the frr is:", far, frr)


    '''
    print("the fpr is:", fpr)
    print("the tpr is:", tpr)
    print("the thresholds is:", thresholds)
    '''
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    thresh = interp1d(fpr, thresholds)(eer)  # Calculated threshold, not needed for score
    eer = eer * 100
    #print("#################the far, the frr, the eer, the f1 is:", far, frr, eer, f1)
    if return_threshold:
        return eer, thresh
    else:
        return far, frr, eer, f1

def get_f1(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_w(logger, percentage_of_outliers, prediction, threshold, gt_inlier):
    y = np.greater(prediction, threshold)

    gt_outlier = np.logical_not(gt_inlier)

    true_positive = np.sum(np.logical_and(y, gt_inlier))
    true_negative = np.sum(np.logical_and(np.logical_not(y), gt_outlier))
    false_positive = np.sum(np.logical_and(y, gt_outlier))
    false_negative = np.sum(np.logical_and(np.logical_not(y), gt_inlier))
    total_count = true_positive + true_negative + false_positive + false_negative
    #print("the true_positive, the true_negative, the false_positive, the false_negative is:", true_positive, true_negative, false_positive, false_negative)
    accuracy = 100 * (true_positive + true_negative) / total_count
    far = 100 * false_positive/(true_negative+false_positive)
    frr = 100 * false_negative/(false_negative + true_positive)
    y_true = gt_inlier
    y_scores = prediction

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0

    m_far,m_frr, m_eer,m_f1 = utils_eer(y_true, y_scores, return_threshold=False)
    '''
    logger.info("Percentage %f" % percentage_of_outliers)
    logger.info("Accuracy %f" % accuracy)
    f1 = get_f1(true_positive, false_positive, false_negative)
    logger.info("F1 %f" % get_f1(true_positive, false_positive, false_negative))
    logger.info("AUC %f" % auc)
    '''
    #print("\nthe far, the frr, the eer is:", far, frr, eer*100)
    print("Percentage %f" % percentage_of_outliers)
    print("Accuracy %f" % accuracy)
    f1 = get_f1(true_positive, false_positive, false_negative)
    print("F1 %f" % get_f1(true_positive, false_positive, false_negative))
    print("AUC %f" % auc)
    
    # return dict(auc=auc, f1=f1)

    # inliers
    X1 = [x[1] for x in zip(gt_inlier, prediction) if x[0]]

    # outliers
    Y1 = [x[1] for x in zip(gt_inlier, prediction) if not x[0]]

    minP = min(prediction) - 1
    maxP = max(prediction) + 1

    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tpr = np.sum(np.greater_equal(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

    logger.info("tpr: %f" % clothest_tpr)
    logger.info("fpr95: %f" % fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    error = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tpr = np.sum(np.less(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        error = np.minimum(error, (tpr + fpr) / 2.0)

    logger.info("Detection error: %f" % error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tp = np.sum(np.greater_equal(X1, threshold))
        fp = np.sum(np.greater_equal(Y1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision

    logger.info("auprin: %f" % auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    minP, maxP = -maxP, -minP
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tp = np.sum(np.greater_equal(Y1, threshold))
        fp = np.sum(np.greater_equal(X1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision

    logger.info("auprout: %f" % auprout)
    '''
    with open(os.path.join("results.txt"), "a") as file:
        file.write(
            "Class: %s\n Percentage: %d\n"
            "Error: %f\n F1: %f\n AUC: %f\nfpr95: %f"
            "\nDetection: %f\nauprin: %f\nauprout: %f\n\n" %
            ("_".join([str(x) for x in inliner_classes]), percentage_of_outliers, error, f1, auc, fpr95, error, auprin, auprout))
    '''
    #return dict(auc=auc, f1=f1, fpr95=fpr95, error=error, auprin=auprin, auprout=auprout)
    return m_far, m_frr, m_eer, m_f1, auc
