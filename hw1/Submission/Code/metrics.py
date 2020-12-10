def get_metrics(y_true, y_pred):
    """
    Function for calculating the F1 score

    Params
    ------
    y_true  : the true labels shaped (N, C), 
              N is the number of datapoints
              C is the number of classes
    y_pred  : the predicted labels, same shape
              as y_true

    Return
    ------
    score   : the F1 score

    """

    # for now, just assume there are 2 classes

    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn)/len(y_true)
    precision = 0 if (tp + fp) == 0 else tp/(tp + fp)
    recall = 0 if (tp + fn) == 0 else tp/(tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2*((precision*recall)/(precision+recall))
    return accuracy, precision, recall, f1
