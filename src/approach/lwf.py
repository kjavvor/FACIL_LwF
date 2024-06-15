import torch
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import traceback

import sys
sys.path.append('D:\studia\zzsn\ZZSN\FACIL_LwF\src')

from approach.calibration1 import PlattScaling, IsotonicCalibration, EnsembleTemperatureScaling
from approach.incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, T=2, calibrate=1, calibration_method=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.calibration_method = calibration_method
        self.calibrate = calibrate
        self.calibrator = None 
        if self.calibrate == 1:
            if self.calibration_method == 'temperature':
                self.calibrator = EnsembleTemperatureScaling(num_models=20)
            elif self.calibration_method == 'platt':
                self.calibrator = PlattScaling()
            elif self.calibration_method == 'isotonic':
                self.calibrator = IsotonicCalibration()
            else:
                self.calibrator = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
    
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        
            
    def post_train_process(self, t, trn_loader, val_loader):
        """Handle post-training including saving the model and calibrating."""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        print(f"Model_old updated after task {t}")

        print("Training with calibration" if self.calibrate else "Training without calibration")
        # Collect logits and labels from the validation set for calibration
        if self.calibrator and self.calibration_method:
            logits, labels = self.collect_logits_labels(val_loader)
            if self.calibration_method == 'temperature':
                self.calibrator.fit(logits, labels.long(), t)
            elif self.calibration_method == 'platt':
                self.calibrator.fit(logits.cpu().numpy(), labels.cpu().numpy(), t)
            elif self.calibration_method == 'isotonic':
                self.calibrator.fit(logits.cpu().numpy(), labels.cpu().numpy(), t)

        super().post_train_process(t, trn_loader, val_loader)


    def collect_logits_labels(self, loader):
        """Collect logits and labels from the loader for calibration purposes."""
        self.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.model(images)

                # Check if logits are in a list and unwrap if necessary
                if isinstance(logits, list) and all(isinstance(l, torch.Tensor) for l in logits):
                    logits = logits[0]  # Assuming there's only one tensor in the list

                all_logits.append(logits.detach())
                all_labels.append(targets.detach())

        if len(all_logits) == 0:
            return torch.tensor([]), torch.tensor([])

        return torch.cat(all_logits), torch.cat(all_labels)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            outputs_old = None
            if t > 0 and self.model_old is not None:
                outputs_old = self.model_old(images)
            if outputs_old is None and t > 0:
                raise ValueError(f"Outputs_old was not generated even though t={t} and model_old is set")
            loss = self.criterion(t, outputs, targets, outputs_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Evaluation without applying real-time calibration; used during training."""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))

                # Forward current model
                logits = self.model(images)

                # Debugging information
                print(f"Task {t}, Logits from model: {logits[:5]}")
                print(f"Task {t}, Targets: {targets[:5]}")

                # Compute loss using logits (calibration does not affect loss calculation)
                loss = self.criterion(t, logits, targets, targets_old)

                # Apply calibration if available and fitted
                if self.calibrator and self.calibrator.is_fitted(t):
                    print(f"***CALIBRATION MODE*** Task ID: {t}")
                    probabilities = self.calibrator.predict_proba(logits, t, device=self.device)
                    for i, prob in enumerate(probabilities):
                        print(f"Probability {i} after calibration: {prob[:5]}")
                    hits_taw, hits_tag = self.calculate_metrics(probabilities, targets)
                else:
                    hits_taw, hits_tag = self.calculate_metrics(logits, targets)

                # Accumulate metrics
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

                print(f"Task {t}, Hits TAw: {hits_taw.sum().item()}, Hits TAg: {hits_tag.sum().item()}")

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        if t > 0 and outputs_old is None:
            print(f"Debug: t={t}, outputs_old is None, model_old is {'set' if self.model_old else 'not set'}")
            raise ValueError("Outputs_old must be initialized for task index > 0")
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        targets = targets.long()

        # print("Debugging Info:")
        # print(f"Outputs tensor shape: {outputs.shape}")
        # print(f"Targets shape: {targets.shape}")
        # print(f"Outputs tensor type: {outputs.dtype}, device: {outputs.device}")
        # print(f"Targets type: {targets.dtype}, device: {targets.device}")

        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
