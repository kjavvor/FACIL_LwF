# import torch
# import torch.nn.functional as F
# from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import IsotonicRegression


# class PlattScaling:
#     def __init__(self):
#         self.model = LogisticRegression()

#     def fit(self, logits, labels):
#         logits = logits.reshape(-1, 1)
#         self.model.fit(logits, labels)

#     def predict_proba(self, logits):
#         logits = logits.reshape(-1, 1)
#         return self.model.predict_proba(logits)[:, 1]

# class IsotonicCalibration:
#     def __init__(self):
#         self.model = IsotonicRegression(out_of_bounds='clip')

#     def fit(self, logits, labels):
#         self.model.fit(logits, labels)

#     def predict_proba(self, logits):
#         return self.model.transform(logits)

# class TemperatureScaling:
#     def __init__(self, temperature=1.0):
#         self.temperature = torch.nn.Parameter(torch.ones(1) * temperature)

#     def set_temperature(self, logits, labels):
#         logits = torch.cat([logit.unsqueeze(0) for logit in logits])
#         labels = torch.cat([label.unsqueeze(0) for label in labels])
#         self.temperature = self.optimize_temperature(logits, labels)

#     def optimize_temperature(self, logits, labels):
#         nll_criterion = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')
#         def eval():
#             optimizer.zero_grad()
#             loss = nll_criterion(logits / self.temperature, labels)
#             loss.backward()
#             return loss
#         optimizer.step(eval)
#         return self.temperature

#     def predict_proba(self, logits):
#         return F.softmax(logits / self.temperature, dim=1)
