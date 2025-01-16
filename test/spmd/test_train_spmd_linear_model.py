import argparse
from contextlib import contextmanager
import os
import sys
import unittest

import torch

import test_xla_sharding_base

parent_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_folder)
from utils.train_spmd_linear_model import train_and_evaluate

# CPU does not support optimization barriers, and hence we use this to disable
# the gradient checkpointing A/B test run for it.
SKIP_GRADIENT_CHECKPOINTING: bool = False


@contextmanager
def extended_argv(args):
  original_argv = sys.argv[:]
  sys.argv.extend(args)
  try:
    yield
  finally:
    sys.argv = original_argv


class TestSPMDLinearModel(test_xla_sharding_base.XlaShardingTest):

  def test_basic(self):
    print('Training loop with baseline')
    with extended_argv([]):
      baseline_losses, baseline_result = train_and_evaluate()
    # Verify that the model losses are not zero.
    assert all(loss != 0 for loss in baseline_losses)
    # Verify that the model produces non-zero outputs.
    assert not torch.any(baseline_result == 0)

    if not SKIP_GRADIENT_CHECKPOINTING:
      print('Training loop with gradient checkpointing')
      with extended_argv(['--use_gradient_checkpointing']):
        checkpointing_losses, checkpointing_result = train_and_evaluate()
        # Verify that the runs match with and without checkpointing.
        assert torch.allclose(
            baseline_result, checkpointing_result, rtol=1e-06, atol=1e-09)
        assert all(
            torch.allclose(baseline_loss, checkpointing_loss)
            for baseline_loss, checkpointing_loss in zip(
                baseline_losses, checkpointing_losses))


class TestSPMDLinearModelGradientAccumulation(
    test_xla_sharding_base.XlaShardingTest):

  def test_gradient_accumulation_matches(self):
    """Verify that gradient accumulation produces the same results and losses
       with and without the XLA `While` op.
    """

    COMMON_GRAD_ACC_ARGS = ["--gradient_accumulation_steps", "8"]
    print('Training loop with traditional gradient accumulation')
    with extended_argv(COMMON_GRAD_ACC_ARGS):
      baseline_grad_acc_losses, baseline_grad_acc_result = train_and_evaluate()

    print('Training loop with XLA\'s `While` gradient accumulation')
    with extended_argv(COMMON_GRAD_ACC_ARGS +
                       ["--use_gradient_accumulation_loop"]):
      loop_grad_acc_losses, loop_grad_acc_result = train_and_evaluate()

    # Verify that the model losses are not zero, and that the runs match.
    assert all(loss != 0 for loss in baseline_grad_acc_losses)
    assert all(
        torch.allclose(baseline_loss, checkpointing_loss)
        for baseline_loss, checkpointing_loss in zip(baseline_grad_acc_losses,
                                                     loop_grad_acc_losses))
    # Verify that the model produces non-zero outputs, and that the runs match.
    assert not torch.any(baseline_grad_acc_result == 0)
    assert torch.allclose(
        baseline_grad_acc_result, loop_grad_acc_result, rtol=1e-06, atol=1e-09)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Relevant parser for the gradient checkpointing basic coverage.
  parser.add_argument('--skip-gradient-checkpointing', action='store_true')
  parsed_args, remaining_argv = parser.parse_known_args()
  SKIP_GRADIENT_CHECKPOINTING = parsed_args.skip_gradient_checkpointing
  test = unittest.main(argv=[sys.argv[0]] + remaining_argv)
  sys.exit(0 if test.result.wasSuccessful() else 1)
