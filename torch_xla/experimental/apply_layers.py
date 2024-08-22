from typing import Iterable

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from torch_xla.experimental.scan import scan


def apply_layers(layers: Iterable[torch.nn.Module], input_data):
  """Applies each layer in `layers` to `input_data` sequentially.

  `input_data` is provided as input to the first layer in `layers`. The output of one
  layer is provided as input to next layer. This function is equivalent to

    sequential = torch.nn.Sequential(layers)
    sequential(input_data)

  This function can be faster to compile since it reuses the XLA computation of the
  first layer to perform the computation of all other layers.
  """
  # Handle empty layers case.
  try:
    next(iter(layers))
  except StopIteration:
    return input_data

  # Extract and stack the parameters into a pytree.
  params = [_extract_weights_dict(layer) for layer in layers]
  stacked_params = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                            *params)

  # Use the first layer as the example/template layer.
  from copy import deepcopy
  example_layer = deepcopy(next(iter(layers)))

  # Hollow out the weights and biases in the example layer.
  example_layer = example_layer.to_empty(device=None)

  # Define the function to apply at each step
  def one_layer(carry, params):
    # Apply the current layer's weights and biases to the example layer,
    # then run the resulting layer.
    _apply_weights_dict(example_layer, params)
    # TODO(yifeit): it should be possible to return `None` as opposed to
    # `example_layer(carry) * 0`, for additional clarity. There is no extra
    # computation since we discard `ys` right after.
    return example_layer(carry), example_layer(carry) * 0

  final_carry, _ = scan(one_layer, input_data, stacked_params)

  return final_carry


def _extract_weights_dict(module: nn.Module):
  """
  Extracts the parameters (weights and biases) from a PyTorch module and
  stores them in a dictionary.
  """
  weights_dict = {
      name: param.clone() for name, param in module.named_parameters()
  }
  return weights_dict


def _apply_weights_dict(module: nn.Module, weights_dict):
  """
  Re-applies the weights and biases from the dictionary back to the PyTorch module.
  """
  for name, param in module.named_parameters():
    if name in weights_dict:
      torch.utils.swap_tensors(param, weights_dict[name].clone())
