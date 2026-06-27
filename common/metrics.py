import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def print_metrics(y_true, y_pred, y_prob):
    print("Classification Report")
    print("-" * 60)
    print(classification_report(y_true, y_pred, digits=4))

    print()

    print("Summary")
    print("-" * 60)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


def plot_roc_curve(y_true, y_prob):
    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
    )

    plt.title("ROC Curve")

    plt.tight_layout()

    plt.show()


def evaluate_binary_classifier(y_true, y_pred, y_prob):
    print_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    print()

    plot_confusion_matrix(
        y_true,
        y_pred,
    )

    plot_roc_curve(
        y_true,
        y_prob,
    )
