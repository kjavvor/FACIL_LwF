import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelBinarizer

class PlattScaling:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, logits, labels):
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        return self.model.predict_proba(logits)

class IsotonicCalibration:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, logits, labels):
        logits = logits.max(axis=1)
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        logits = logits.max(axis=1)
        probs = self.model.transform(logits)
        return probs

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, logits, labels):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

    def predict_proba(self, logits):
        calibrated_logits = self.temperature_scale(logits)
        return F.softmax(calibrated_logits, dim=1)
