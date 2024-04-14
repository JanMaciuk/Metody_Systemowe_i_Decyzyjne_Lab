from typing import List, Tuple


def get_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists. 

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    # Protect from obviously invalid input
    if len(y_true) != len(y_pred):
        raise ValueError("Invalid input shapes!")
    if (num_classes < max(max(y_true), max(y_pred))) or (num_classes < 0):
        raise ValueError("Invalid prediction classes!")

    
    # Zero matrix, size is equal to the number of classes (square matrix)
    confusion_matrix = [[0] * num_classes for i in range(num_classes)]
    
    for actual, predicted in zip(y_true, y_pred):
        confusion_matrix[actual][predicted] += 1
    
    return confusion_matrix


def get_quality_factors(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive 
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of input lists don't match.")
    
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative
    TP = 0  # True Positive
    
    for actual, predicted in zip(y_true, y_pred):
        if actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1
        elif actual == 1 and predicted == 1:
            TP += 1
    
    return TN, FP, FN, TP


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    return ((TP + TN) / len(y_true))    # length of true and pred is validated to be the same in get_quality_factors


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    return (TP / (TP + FP))


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    return (TP / (TP + FN))


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)
    return (2*precision_score(y_true, y_pred)*recall_score(y_true, y_pred)) / (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))

# all scores calculated with formulas from https://www.geeksforgeeks.org/confusion-matrix-machine-learning/