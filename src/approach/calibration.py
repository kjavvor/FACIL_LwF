import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np


class PlattScaling:
    def __init__(self, num_classes):
        # Initialize a logistic regression model for each class
        self.models = [LogisticRegression() for _ in range(num_classes)]

    def fit(self, logits, labels):
        # Fit a logistic regression model for each class
        for i in range(len(self.models)):
            class_logits = logits[:, i].reshape(-1, 1)
            class_labels = (labels == i).astype(int)  # Binary labels for each class
            self.models[i].fit(class_logits, class_labels)

    def predict_proba(self, logits):
        # Predict probabilities for each class
        probas = np.array([model.predict_proba(logits[:, i].reshape(-1, 1))[:, 1]
                           for i, model in enumerate(self.models)]).T
        return probas
    
class IsotonicCalibration:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, logits, labels):
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        return self.model.transform(logits)

class TemperatureScaling:
    def __init__(self, temperature=1.0):
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature)

    def set_temperature(self, logits, labels):
        logits = torch.cat([logit.unsqueeze(0) for logit in logits])
        labels = torch.cat([label.unsqueeze(0) for label in labels])
        self.temperature = self.optimize_temperature(logits, labels)

    def optimize_temperature(self, logits, labels):
        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        return self.temperature

    def predict_proba(self, logits):
        return F.softmax(logits / self.temperature, dim=1)