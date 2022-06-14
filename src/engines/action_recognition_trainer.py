from torchmetrics import Accuracy, MetricCollection

from dlhammer.nested import nested_call
import dlhammer.distributed as du
from dlhammer.loss_meters import LossMeter
from dlhammer.base_trainer import BaseTrainer

from .builder import TRAINERS as registry


@registry.register('action_recognition')
class ActionRecognitionTrainer(BaseTrainer):
    """Trainer for action recognition tasks with video-level supervision"""

    def __init__(self, *args, **kwargs):
        super(ActionRecognitionTrainer, self).__init__(*args, **kwargs)

        eval_meters_obj = MetricCollection({
            'top1': Accuracy(top_k=1).to(self.device),
            'top5': Accuracy(top_k=5).to(self.device),
        })
        self.eval_meters = {}
        self.loss_meters = {}
        for phase in self.cfg.SOLVER.PHASES:
            self.eval_meters[phase] = eval_meters_obj.clone(prefix=f'{phase}/')
            self.loss_meters[phase] = LossMeter()

    def reduce_and_cal_meters(self, phase, loss, logits, labels):
        [loss] = du.all_reduce([loss])
        batch_logs = self.eval_meters[phase](logits.softmax(-1), labels)
        batch_logs = {
            key: du.all_reduce([value])[0].item() for key, value in batch_logs.items()
        }
        return loss, batch_logs

    def train_step(self, batch):
        inputs, labels = batch

        logits = self.forward(inputs)

        loss = self.criterion['SCE'](logits, labels)

        nested_call(self.optimizer, 'zero_grad')
        loss.backward()
        nested_call(self.optimizer, 'step')
        nested_call(self.scheduler, 'step')

        return self.reduce_and_cal_meters(self.phase, loss, logits, labels)

    def test_step(self, batch):
        inputs, labels = batch

        logits = self.forward(inputs)

        loss = self.criterion['SCE'](logits, labels)

        return self.reduce_and_cal_meters(self.phase, loss, logits, labels)

    def forward(self, x):
        return self.model(x)
