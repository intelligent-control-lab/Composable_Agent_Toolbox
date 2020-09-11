import os
import re
import torch
import yaml
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def to_numpy(var):
  return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
  return Variable(
      torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
  ).type(dtype)

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(
      target_param.data * (1.0 - tau) + param.data * tau
    )

def hard_update(target, source):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(param.data)

def load_experiment_settings(experiment_settings, filepath=None):
  loader = yaml.SafeLoader
  loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

  all_settings = {}
  if experiment_settings and experiment_settings[0]:
    for experiment_setting in experiment_settings:
      with open(experiment_setting, 'r') as f:
        settings = yaml.load(f, Loader=loader)
        if settings:
          all_settings.update(settings)

  return all_settings

color2num = dict(
  gray=30,
  red=31,
  green=32,
  yellow=33,
  blue=34,
  magenta=35,
  cyan=36,
  white=37,
  crimson=38
)

def colorize(string, color, bold=False, highlight=False):
  """
  Colorize a string.

  This function was originally written by John Schulman.
  """
  attr = []
  num = color2num[color]
  if highlight: num += 10
  attr.append(str(num))
  if bold: attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def log(msg, color='green'):
    """Print a colorized message to stdout."""
    print(colorize(msg, color, bold=True))
