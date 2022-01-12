
import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.utilities.checks import _input_squeeze


class CamvidAccuracy(Metric):
    
    def __init__(self, void_code=30, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.void_code = void_code
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        preds, target = _input_squeeze(preds, target)
        assert preds.shape == target.shape
        mask = target != self.void_code
        self.correct += torch.sum(preds[mask] == target[mask])
        self.total += target[mask].numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total
