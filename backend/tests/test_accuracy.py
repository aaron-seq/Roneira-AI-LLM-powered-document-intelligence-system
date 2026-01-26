import pytest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np

def test_model_accuracy_matrix():
    """
    Simulates a Confusion Matrix evaluation for an AI component.
    """
    # 1. Ground Truth (Before/Expected Data)
    # 0 = Negative Class (e.g., Document Rejected)
    # 1 = Positive Class (e.g., Document Accepted)
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1] 
    
    # 2. Model Predictions (Actual Output)
    y_pred = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    # Prediction errors: 
    # Index 6: True=0, Pred=1 (False Positive)
    # Index 8: True=1, Pred=0 (False Negative)
    
    # 3. Calculation
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # 4. Assertions (Threshold Validation)
    
    # We expect Accuracy to be 80% (8/10 correct)
    assert accuracy == 0.8, f"Accuracy {accuracy} below expected 0.8"
    
    # We expect Precision to be ~85.7% (6 True Positives / 7 Predicted Positives)
    # TP=6, FP=1 -> 6/7 = 0.857
    assert precision > 0.85
    
    # We expect Recall to be ~83.3% (5 True Positives / 6 Actual Positives)
    # Note: actually in this array:
    # y_true has six 1s.
    # y_pred matched 1s at indices: 1, 3, 5, 7, 9 (5 matches).
    # Missed index 8.
    # So Recall = 5/6 = 0.8333...
    assert recall > 0.8
    
    # Confusion Matrix Shape
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = conf_matrix.ravel()
    assert tn == 3 # Indices 0, 2, 4 derived correctly? y_true 0s are at 0, 2, 4, 6. 
    # y_pred[6] was 1. So 3 TNs.
    assert fp == 1
    assert fn == 1
    assert tp == 5

if __name__ == "__main__":
    test_model_accuracy_matrix()
