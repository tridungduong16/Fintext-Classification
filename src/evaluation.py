from sklearn.metrics import classification_report, confusion_matrix


def evaluate_sentiment(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {"report": report, "confusion_matrix": conf_matrix}
