import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
import torch.optim as optim
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        self.to(logits.device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def step():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(step)

class EnsembleTemperatureScaling:
    def __init__(self, num_models=5):
        self.models = []
        self.num_models = num_models
        self.task_models = []
        self.num_classes = None

    def fit(self, logits, labels, task_id):
        self.task_models = [TemperatureScaling() for _ in range(self.num_models)]
        subsets = torch.chunk(logits, self.num_models, dim=0)
        label_subsets = torch.chunk(labels, self.num_models, dim=0)

        for model, subset, label_subset in zip(self.task_models, subsets, label_subsets):
            print(f"Fitting model with logits shape {subset.shape} and labels shape {label_subset.shape}")
            print(f"Label range: {label_subset.min()} to {label_subset.max()}")
            model.fit(subset, label_subset)

        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = self.task_models

        print(f"Number of models for task {task_id}: {len(self.task_models)}")
        print(f"Total number of models: {sum(len(models) for models in self.models if models is not None)}")

    def predict_proba(self, logits, task_id, device):
        # Apply each model's scaling and average the results
        if self.task_models is None:
            return False

        task_models = self.models[task_id]
        mean_probas_list = []
        with torch.no_grad():
            for logit in logits:
                scaled_logit = [model(logit) for model in self.models]
                scaled_probs = [nn.functional.softmax(l, dim=1) for l in scaled_logit]
                mean_probs = torch.mean(torch.stack(scaled_probs), dim=0)
                mean_probas_list.append(mean_probs.to(device))
        return mean_probas_list

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

class PlattScaling:
    def __init__(self):
        self.models = []  # This will now be a list of lists, each containing models for a specific task
        self.num_classes = None
        self.task_models = None

    def fit(self, logits, labels, task_id):
        """Fit Platt scaling models to logits for a specific task identified by task_id."""
        self.num_classes = logits.shape[1]
        self.task_models = [LogisticRegression() for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            class_logits = logits[:, i].reshape(-1, 1)
            # print("Labels:", labels)
            class_labels = (labels == i + task_id * self.num_classes).astype(int)
            # print("Class labels:", class_labels)
            unique_labels = np.unique(class_labels)
            # print(f"Task {task_id}, Class {i}, Unique labels: {unique_labels}")
            if len(unique_labels) > 1:
                self.task_models[i].fit(class_logits, class_labels)
            else:
                print(f"Task {task_id}, Class {i}, Single unique label: {unique_labels}")
                self.task_models[i] = None  # Handle single class by not fitting model
        # Ensure we have enough lists to cover all tasks up to task_id
        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = self.task_models

        # print(f"Number of models for task {task_id}: {len(self.task_models)}")
        # print("Len models: ", len(self.models))
        # print(f"Total number of models: {sum(len(models) for models in self.models if models is not None)}")
        # print(f"Set num_classes to: {self.num_classes}")
        # print(f"Models for task {task_id}: {self.task_models}")

    def predict_proba(self, logits, task_id, device):
        if not self.is_fitted(task_id):
            raise RuntimeError(f"Models for task {task_id} have not been fitted yet.")

        task_models = self.models[task_id]
        # print(f'Models', self.models)
        # print(f'task models', task_models)
        probas_list = []

        for logit in logits:
            if logit.is_cuda:
                logit = logit.cpu()  # Move to CPU for compatibility with sklearn models

            logit_np = logit.numpy()  # Convert to numpy array for logistic regression

            probas = np.zeros_like(logit_np)
            for i, model in enumerate(task_models):
                if model is not None:
                    probas[:, i] = model.predict_proba(logit_np[:, i].reshape(-1, 1))[:, 1]
                else:
                    print(f"Task {task_id}, Model {i} is None")
                    probas[:, i] = 1.0  # Default to 100% probability if only one class was available during fit

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
        self.models = []  # This will now be a list of lists, each containing models for a specific task
        self.num_classes = None
        self.task_models = None

    def fit(self, logits, labels, task_id):
        """Fit isotonic models to logits for a specific task identified by task_id."""
        self.num_classes = logits.shape[1]
        self.task_models = [IsotonicRegression(out_of_bounds='clip') for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.task_models[i].fit(logits[:, i], (labels == i + task_id * self.num_classes).astype(float))
        # Ensure we have enough lists to cover all tasks up to task_id
        while len(self.models) <= task_id:
            self.models.append(None)
        self.models[task_id] = self.task_models

        # Print the number of models per task and the total number of models
        print(f"Number of models for task {task_id}: {len(self.task_models)}")
        print(f"Total number of models: {sum(len(models) for models in self.models if models is not None)}")

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

