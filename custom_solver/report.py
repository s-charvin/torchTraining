from sklearn.metrics import classification_report as sk_classification_report
from sklearn import metrics


def classification_report(y_true, y_pred, target_names=None, digits=6):
    report = sk_classification_report(
        y_true, y_pred, digits=digits, target_names=target_names, zero_division=1)
    report_dict = sk_classification_report(
        y_true, y_pred, digits=digits, target_names=target_names, output_dict=True, zero_division=1)
    matrix = metrics.confusion_matrix(y_true, y_pred)

    UA = report_dict['macro avg']['recall'] * 100
    macro_f1 = report_dict['macro avg']['f1-score'] * 100
    WA = report_dict['accuracy'] * 100
    w_f1 = report_dict['weighted avg']['f1-score'] * 100
    ACC = (WA + UA)/2
    return WA, UA, ACC, macro_f1, w_f1, report, matrix
