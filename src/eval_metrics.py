from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns a dictionary of various performance metrics.
    """
    # Ensure labels are specified to cover both classes 0 and 1
    labels = [0, 1]  # Adjust this list if you have more than two classes

    # Existing metrics
    metrics = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=labels)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate confusion matrix with defined labels
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]

    # Calculate TNR and TPR
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # Avoid division by zero
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # TPR is same as recall for binary classification

    return {
        "precision": metrics[0],
        "recall": metrics[1],  # Recall is equivalent to TPR in binary classification
        "F1-Score": metrics[2],
        "accuracy": accuracy,
        "TNR": TNR,
        "TPR": TPR  # Added TPR for clarity
    }