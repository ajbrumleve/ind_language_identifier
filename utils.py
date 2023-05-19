from sklearn.metrics import classification_report

from logger import *

@log_to_file("domain_class_reports.txt")
def get_domain_class_report(result, pred):
    report = classification_report(result, pred, digits=5)
    # Print the classification report
    print("Classification Report:")
    print(report)
    return report

@log_to_file("predict_language_reports.txt")
def get_predict_language_report(result, pred):
    report = classification_report(result, pred, digits=5)
    # Print the classification report
    print("Classification Report:")
    print(report)
    return report