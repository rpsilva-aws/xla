import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class TrainResNetXLADDP(TrainResNetBase):

  def __init__(self):
    super().__init__()
    # below code is commented out because in this example we used a fake data
    # loader that does not take sampler. However this logic is needed if you
    # want each process to handle different parts of the data.
    '''
    train_sampler = None
    if xr.world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xr.world_size(),
          rank=xr.global_ordinal(),
          shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        prefetch_factor=FLAGS.prefetch_factor)
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    '''

  def run_optimizer(self):
    # optimizer_step will call `optimizer.step()` and all_reduce the gradident
    xm.optimizer_step(self.optimizer)


def _mp_fn(index):
  xla_ddp = TrainResNetXLADDP()
  xla_ddp.start_training()


if __name__ == '__main__':
  print(
      'consider using train_resnet_spmd_data_parallel.py instead to get better performance'
  )
  torch_xla.launch(_mp_fn, args=())
