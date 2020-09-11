import os
from spinningup.spinup.utils.logx import EpochLogger, colorize
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger(EpochLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_dir = os.path.join(self.output_dir, 'tensorboard')
        self.writer = SummaryWriter(log_dir=log_dir)
        print(colorize(f"Logging tb event to {log_dir}", 'green', bold=True))

    @property
    def __name__(self):
        return "TensorboardLogger"

    def log_epoch_stats(self, epoch, group, key, val=None, with_min_and_max=False, average_only=False):
      key, stats = super().log_tabular(key, val, with_min_and_max, average_only)

      if isinstance(stats, tuple):
        self.writer.add_scalar('{}/avg_{}'.format(group, key), stats[0], epoch)
        if len(stats) == 2:
          self.writer.add_scalar('{}/std_{}'.format(group, key), stats[1], epoch)
        if len(stats) > 2:
          self.writer.add_scalar('{}/min_{}'.format(group, key), stats[2], epoch)
          self.writer.add_scalar('{}/max_{}'.format(group, key), stats[3], epoch)
      else:
        self.writer.add_scalar('{}/{}'.format(group, key), val, epoch)

