import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted_flag = False

    def fit(self, logits, labels, num_classes):
        logits = np.asarray(logits)
        labels = np.asarray(labels).astype(int)
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Labels are out of bounds. Should be in range [0, {num_classes - 1}]")

        def loss_fn(temp):
            temp_logits = logits / temp
            temp_logits = np.hstack([1 - temp_logits, temp_logits])  # ensure shape (N, 2) for binary classification
            loss = F.cross_entropy(torch.tensor(temp_logits, dtype=torch.float32),
                                   torch.tensor(labels, dtype=torch.long))
            return loss.item()

        res = minimize(loss_fn, x0=[self.temperature], bounds=[(0.5, 5.0)], method='L-BFGS-B')
        self.temperature = res.x[0]
        self.is_fitted_flag = True

    def transform(self, logits):
        logits = np.asarray(logits)
        return logits / self.temperature

    def predict_proba(self, logits):
        temp_logits = self.transform(logits)
        temp_logits = np.hstack([1 - temp_logits, temp_logits])
        exp_logits = np.exp(temp_logits - np.max(temp_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def is_fitted(self):
        return self.is_fitted_flag


class EnsembleTemperatureScaling:
    def __init__(self):
        self.models = []
        self.num_classes = None

    def fit(self, logits, labels, task_id):
        """Fit calibrator fro a specific task"""
        self.num_classes = logits.shape[1]
        task_models = [TemperatureScaling() for _ in range(self.num_classes)]

        for i in range(self.num_classes):
            class_logits = logits[:, [i]]  # Ensure logits have shape (N, 1)
            class_labels = (labels == i + task_id * self.num_classes).astype(int)
            unique_labels = np.unique(class_labels)

            if class_labels.min() < 0 or class_labels.max() > 1:
                raise ValueError("Binary class labels should be 0 or 1.")

            if len(unique_labels) > 1:
                task_models[i].fit(class_logits, class_labels, 2)
            else:
                task_models[i] = None

        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = task_models

    def predict_proba(self, logits, task_id, device):
        """"Predict probabilities for logits for a specific task identified by task_id."""
        if not self.is_fitted(task_id):
            raise RuntimeError(f"Models for task {task_id} have not been fitted yet.")

        task_models = self.models[task_id]
        probas_list = []
        for logit in logits:
            if logit.is_cuda:
                logit = logit.cpu()

            logit_np = logit.numpy()
            probas = np.zeros((logit_np.shape[0], len(task_models)))
            for i, model in enumerate(task_models):
                if model is not None:
                    probas[:, i] = model.predict_proba(logit_np[:, [i]])[:, 1]
                else:
                    probas[:, i] = 1.0
            probas_tensor = torch.tensor(probas, dtype=torch.float32, device=device)
            probas_list.append(probas_tensor)
        return probas_list

    def is_fitted(self, task_id):
        if len(self.models) < task_id + 1:
            return False

        for model in self.models[task_id]:
            if model is not None:
                if not model.is_fitted():
                    return False
        return True


class PlattScaling:
    def __init__(self):
        self.models = []
        self.num_classes = None
        self.task_models = None

    def fit(self, logits, labels, task_id):
        """Fit Platt scaling models to logits for a specific task identified by task_id."""
        self.num_classes = logits.shape[1]
        self.task_models = [LogisticRegression() for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            class_logits = logits[:, i].reshape(-1, 1)
            class_labels = (labels == i + task_id * self.num_classes).astype(int)
            unique_labels = np.unique(class_labels)
            if len(unique_labels) > 1:
                self.task_models[i].fit(class_logits, class_labels)
            else:
                print(f"Task {task_id}, Class {i}, Single unique label: {unique_labels}")
                self.task_models[i] = None
        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = self.task_models


    def predict_proba(self, logits, task_id, device):
        """"Predict probabilities for logits for a specific task identified by task_id."""
        if not self.is_fitted(task_id):
            raise RuntimeError(f"Models for task {task_id} have not been fitted yet.")

        task_models = self.models[task_id]
        probas_list = []
        for logit in logits:
            if logit.is_cuda:
                logit = logit.cpu()

            logit_np = logit.numpy()

            probas = np.zeros_like(logit_np)
            for i, model in enumerate(task_models):
                if model is not None:
                    probas[:, i] = model.predict_proba(logit_np[:, i].reshape(-1, 1))[:, 1]
                else:
                    print(f"Task {task_id}, Model {i} is None")
                    probas[:, i] = 1.0

            probas_tensor = torch.tensor(probas, dtype=torch.float32, device=device)
            probas_list.append(probas_tensor)

        return probas_list

    def is_fitted(self, task_id):
        """Check if models for a specific task are fitted."""
        if len(self.models) < task_id + 1:
            return False

        for model in self.models[task_id]:
            if model is not None:
                try:
                    check_is_fitted(model)
                except NotFittedError:
                    print('NO')
                    return False
        return True

class IsotonicCalibration:
    def __init__(self):
        self.models = []
        self.num_classes = None
        self.task_models = None

    def fit(self, logits, labels, task_id):
        """Fit isotonic models to logits for a specific task identified by task_id."""
        self.num_classes = logits.shape[1]
        self.task_models = [IsotonicRegression(out_of_bounds='clip') for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.task_models[i].fit(logits[:, i], (labels == i + task_id * self.num_classes).astype(float))
        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = self.task_models

    def is_fitted(self, task_id):
        """Check if models for a specific task are fitted."""
        if len(self.models) < task_id + 1:
            return False

        for model in self.models[task_id]:
            if model is not None:
                try:
                    check_is_fitted(model)
                except NotFittedError:
                    print('NO')
                    return False
        return True

    def predict_proba(self, logits, task_id, device):
        """Predict probabilities based on given logits for a specific task identified by task_id."""
        if not self.is_fitted(task_id):
            raise RuntimeError(f"Models for task {task_id} have not been fitted yet.")

        task_models = self.models[task_id]
        probas_list = []

        for logit in logits:
            if logit.is_cuda:
                logit = logit.cpu()  # Move to CPU for compatibility with sklearn models

            logit_np = logit.numpy()  # Convert to numpy array for isotonic regression

            probas = np.column_stack([model.transform(logit_np[:, i]) for i, model in enumerate(task_models)])

            probas_tensor = torch.tensor(probas, dtype=torch.float32, device=device)
            probas_list.append(probas_tensor)

        return probas_list
