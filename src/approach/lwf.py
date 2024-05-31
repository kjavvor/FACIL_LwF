import torch
from copy import deepcopy
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from calibration import PlattScaling, IsotonicCalibration, TemperatureScaling

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.calibrator = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        super().train_loop(t, trn_loader, val_loader)
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        self.calibrate(trn_loader, method='platt')  # Można zmienić na 'temperature' lub 'isotonic'

    def calibrate(self, data_loader, method='platt'):
        self.model.eval()
        logits, labels = self.collect_logits_labels(data_loader)
        
        # Debugging information
        print(f'Rozmiar logits: {logits.size()}')
        print(f'Rozmiar labels: {labels.size()}')

        # Store original accuracy before calibration
        original_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f'Original accuracy before calibration: {original_acc * 100:.2f}%')

        if method == 'platt':
            self.calibrator = PlattScaling()
        elif method == 'isotonic':
            self.calibrator = IsotonicCalibration()
        elif method == 'temperature':
            self.calibrator = TemperatureScaling()
            self.calibrator.set_temperature(logits, labels)
            calibrated_logits = self.calibrator.temperature_scale(logits)
            calibrated_probs = torch.nn.functional.softmax(calibrated_logits, dim=1).cpu().numpy()
        else:
            raise ValueError("Invalid calibration method")

        if method in ['platt', 'isotonic']:
            self.calibrator.fit(logits.cpu().numpy(), labels.cpu().numpy())
            calibrated_probs = self.calibrator.predict_proba(logits.cpu().numpy())

        # Debugging information
        print(f'Wymiar calibrated_probs: {calibrated_probs.shape}')

        # Evaluate accuracy after calibration
        calibrated_acc = (torch.tensor(calibrated_probs).argmax(dim=1) == labels).float().mean().item()
        print(f'Calibrated accuracy with {method} method: {calibrated_acc * 100:.2f}%')






    def collect_logits_labels(self, data_loader):
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # Upewnijmy się, że outputs jest tensorem
                if isinstance(outputs, list):
                    outputs = torch.cat(outputs, dim=1)
                logits_list.append(outputs)
                labels_list.append(targets)
        
        # Teraz upewnijmy się, że każdy element w logits_list i labels_list jest tensorami
        logits_list = [logit if isinstance(logit, torch.Tensor) else torch.tensor(logit) for logit in logits_list]
        labels_list = [label if isinstance(label, torch.Tensor) else torch.tensor(label) for label in labels_list]
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Debugowanie rozmiarów logits i labels
        print(f'Rozmiar logits: {logits.size()}')
        print(f'Rozmiar labels: {labels.size()}')

        return logits, labels




    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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
        loss = 0
        if t > 0:
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], (targets - self.model.task_offset[t]).long())
