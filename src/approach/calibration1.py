import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np

class TemperatureScaling:
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1, device=model.device))

    def set_temperature(self, logits, labels):
        logits = logits.data
        labels = labels.data
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)

    def predict_proba(self, logits):
        return F.softmax(logits / self.temperature, dim=1)

class PlattScaling:
    """Platt scaling to calibrate probabilities for multi-class outputs."""
    def __init__(self):
        self.models = []

    def fit(self, logits, labels):
        """Fit logistic regression models for each class based on logits."""
        if logits.ndim != 2:
            raise ValueError("Logits must be a 2D array.")
        num_classes = logits.shape[1]
        self.models = [LogisticRegression() for _ in range(num_classes)]
        for i in range(num_classes):
            class_logits = logits[:, i].reshape(-1, 1)
            class_labels = (labels == i).astype(int)
            self.models[i].fit(class_logits, class_labels)

    def predict_proba(self, logits):
        """Predict class probabilities using the fitted logistic regression models."""
        if not self.models:
            raise RuntimeError("Models have not been fitted yet.")
        if logits.shape[1] != len(self.models):
            raise ValueError("Logit dimensions do not match the number of fitted models.")
        probas = np.array([self.models[i].predict_proba(logits[:, i].reshape(-1, 1))[:, 1]
                           for i in range(len(self.models))]).T
        return probas

class IsotonicCalibration:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, logits, labels):
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        return self.model.transform(logits)