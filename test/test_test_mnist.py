import torch
import torchvision
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import _xla_while_loop
from torch._higher_order_ops.while_loop import while_loop

n_epochs = 3
batch_size_train = 8 # 64
batch_size_test = 10 # 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

### load data
test_loader = xu.SampleGenerator(
    data=(torch.zeros(8, 1, 28,28), torch.zeros(8, dtype=torch.int64)),
    sample_count=1000 // 8 // xm.xrt_world_size())

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print("shape: ", example_data.shape)

### build model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# model.parameters()

class SimpleWithLinearPure(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2).to(xm.xla_device())
    self.bn1 = torch.nn.BatchNorm2d(10).to(xm.xla_device())
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5).to(xm.xla_device())
    self.bn2 = torch.nn.BatchNorm2d(20).to(xm.xla_device())
    self.fc1 = torch.nn.Linear(500, 50).to(xm.xla_device())
    self.fc2 = torch.nn.Linear(50, 10).to(xm.xla_device())
    # self.fc1 = torch.nn.Linear(320, 50).to(xm.xla_device())
    # self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())
    # self.linear2 = torch.nn.Linear(20, 30).to(xm.xla_device())
    # self.fc1 = nn.Linear(320, 50).to(xm.xla_device())
    # self.fc2 = nn.Linear(50, 10).to(xm.xla_device())

  # def forward(self, upper, lower, one_value, x, input_value, output_value):
  def forward(self, x):
    # output_value_real = self.linear(input_value)
    # output_value_real_final = self.linear2(output_value_real)
    # output_value_real_final = self.conv1(input_value) # conv2d
    x = F.relu(F.max_pool2d(self.conv1(x), 2)) # conv2d+mnist-treat
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
    # return x

class SimpleWithLinear(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())
    self.linear2 = torch.nn.Linear(20, 30).to(xm.xla_device())

  def forward(self, upper, lower, one_value, x, input_value, output_value):
    def cond_fn(upper, lower, one_value, x, input_value, output_value):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value):
      new_lower = torch.add(one_value, lower)
      output_value_real = self.linear(input_value)
      output_value_real_final = self.linear2(output_value_real)
      # weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
      # bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
      return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
          one_value, x), input_value.clone(
          ), output_value_real_final # , weight.clone(), bias.clone()

    return while_loop(
        cond_fn, body_fn,
        (upper, lower, one_value, x, input_value, output_value))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = torch.nn.BatchNorm2d(10)
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = torch.nn.BatchNorm2d(20)
    self.fc1 = torch.nn.Linear(320, 50)
    self.fc2 = torch.nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

device = xm.xla_device()
network = MNIST().to(device)
# network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
# loss_fn = nn.NLLLoss()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*20 for i in range(n_epochs + 1)]

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= 20
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, 20,
    100. * correct / 20))
  return test_loss

def new_test():
  simple_with_linear = SimpleWithLinear()
  upper = torch.tensor([52], dtype=torch.int32, device=device)
  lower = torch.tensor([0], dtype=torch.int32, device=device)
  one_value = torch.tensor([1], dtype=torch.int32, device=device)
  init_val = torch.tensor([1], dtype=torch.int32, device=device)
  l_in_0 = torch.rand(10, device=xm.xla_device())
  output_value = torch.zeros([20], dtype=torch.float32, device=device)

  weight_0 = simple_with_linear.linear.weight
  bias_0 = simple_with_linear.linear.bias

  upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
      upper, lower, one_value, init_val, l_in_0, output_value)
  print("finish new_test")

def newnew_test():
  device = xm.xla_device()
  torch.set_grad_enabled(False)

  linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
  # simple_with_linear = SimpleWithLinear()

  def cond_fn(upper, lower, one_value, x, input_value, output_value):
    return lower[0] < upper[0]

  def body_fn(upper, lower, one_value, x, input_value, output_value):
    new_lower = torch.add(one_value, lower)
    output_value = linear_0(input_value)
    weight = linear_0.weight  # not be used actually, initialized as placeholder xlacomputation requirement
    bias = linear_0.bias  # not be used actually, initialized as placeholder xlacomputation requirement
    return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
        one_value, x), input_value.clone(), bias.clone(), weight.clone(
        ), output_value.clone()

  upper = torch.tensor([1], dtype=torch.int32, device=device)
  lower = torch.tensor([0], dtype=torch.int32, device=device)
  one_value = torch.tensor([1], dtype=torch.int32, device=device)
  init_val = torch.tensor([1], dtype=torch.int32, device=device)
  l_in_0 = torch.rand(10, device=xm.xla_device())
  output_value = torch.zeros([20], dtype=torch.float32, device=device)

  upper__, lower__, one_value__, torch_add_res__, input_value__, bias__, weight__, output_value_real__, = while_loop(
      cond_fn, body_fn,
      (upper, lower, one_value, init_val, l_in_0, output_value))
  print("finish newnew_test")

  # # simple_with_linear = SimpleWithLinear()
  # simple_with_linear = SimpleWithLinear()
  # upper = torch.tensor([52], dtype=torch.int32, device=device)
  # lower = torch.tensor([0], dtype=torch.int32, device=device)
  # one_value = torch.tensor([1], dtype=torch.int32, device=device)
  # init_val = torch.tensor([1], dtype=torch.int32, device=device)
  # l_in_0 = torch.rand(10, device=xm.xla_device())
  # output_value = torch.zeros([20], dtype=torch.float32, device=device)
  # weight_0 = simple_with_linear.linear.weight
  # bias_0 = simple_with_linear.linear.bias

  # upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
  #     upper, lower, one_value, init_val, l_in_0, output_value)
  # print("finish new_test")

  # def cond_fn(upper, lower, one_value, x, input_value, output_value):
  #   return lower[0] < upper[0]

  # def body_fn(upper, lower, one_value, x, input_value, output_value):
  #   new_lower = torch.add(one_value, lower)
  #   output_value_real = self.linear(input_value)
  #   output_value_real_final = self.linear2(output_value_real)
  #   return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
  #     one_value, x), input_value.clone(
  #     ), output_value_real_final # , weight.clone(), bias.clone()

  # return while_loop(
  #       cond_fn, body_fn,
  #       (upper, lower, one_value, x, input_value, output_value))

def newnewnew_test():
  device = xm.xla_device()
  torch.set_grad_enabled(False)

  # linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
  # simple_with_linear = SimpleWithLinear()
  simple_with_linear = SimpleWithLinearPure()

  def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
    return lower[0] < upper[0]

  def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
    new_lower = torch.add(one_value, lower)
    output_value = simple_with_linear(input_value)
    # weight = simple_with_linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
    # bias = simple_with_linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
    res = [upper.clone(), new_lower.clone(), one_value.clone(), torch.add(one_value, x), input_value.clone(), output_value.clone()]
    bn_list = []
    # bn_flag = False
    for name, param in simple_with_linear.named_parameters():
      if name[:2]=='bn':
        # bn_flag = True
        # bn_list.insert(-1, param) # dumpicate # continue # skip bn
        bn_list.append(param)
      # else:
        # bn_flag = False

      res.insert(-1, param)

      # if (not bn_flag) and (len(bn_list) !=0): # False
      #   output_value = res[-1]
      #   res = res[:-1] + bn_list #  + res[-1]
      #   res.append(output_value)
      #   bn_list = []
      # # torch.randint(10, carried_input.size(), dtype=carried_input.dtype).to(device))

    ### !!! add still exist bn_list if the last additional_inputs is bn- pre
    # if bn_flag and (len(bn_list) !=0):
    ### !!! add at the tile
    if len(bn_list) !=0:
      output_value = res[-1]
      # res = res[:-1] + bn_list #  + res[-1]
      bn_list.reverse()
      res = res[:-1] + bn_list
      res.append(output_value)
      bn_list = []
      # bn_flag = False

    return tuple(res)
    # return (upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
    #     one_value, x), input_value.clone(), output_value.clone(), simple_with_linear.linear.weight) # bias.clone(), weight.clone(), output_value.clone()

  # print("simple_with_linear weight: ", simple_with_linear.weight)
  # print("simple_with_linear bias: ", simple_with_linear.bias)
  # print("print all things!!!")
  # print(type(simple_with_linear.parameters()))
  # print("simple_with_linear.named_parameters(): ", simple_with_linear.named_parameters())
  # import pdb; pdb.set_trace()

  # for name, param in simple_with_linear.named_parameters():
  #   # print("arrive the loop")
  #   print("name: ", name)
  #   print("param: ", param)

    # if name in ['bias']:
      # print(param.size())

  upper = torch.tensor([50], dtype=torch.int32, device=device)
  lower = torch.tensor([0], dtype=torch.int32, device=device)
  one_value = torch.tensor([1], dtype=torch.int32, device=device)
  init_val = torch.tensor([1], dtype=torch.int32, device=device)
  ### linear 10*20 + 20*30 input&output
  # l_in_0 = torch.rand(10, device=xm.xla_device())
  # output_value = torch.zeros([30], dtype=torch.float32, device=device)
  ### conv2d input&output
  bs=16
  l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
  # c = nn.Conv2d(3,10,kernel_size=5,stride=1,padding=2)
  # out = c(x)
  # print(out.nelement())
  # output_value = torch.zeros([16,10,28,28], dtype=torch.float32, device=device) # conv2d
  # output_value = torch.zeros([16,10,14,14], dtype=torch.float32, device=device) # conv2d+mnist-treat # conv1 + bn1
  # output_value = torch.zeros([16,20,5,5], dtype=torch.float32, device=device) # conv1 + bn1 + conv2
  # output_value = torch.zeros([16,500], dtype=torch.float32, device=device) # conv1 + bn1 + conv2 + bn2 + flatten1
  # output_value = torch.zeros([16,50], dtype=torch.float32, device=device) # conv1 + bn1 + conv2 + bn2 + flatten1 + fc1
  output_value = torch.zeros([16,10], dtype=torch.float32, device=device) # conv1 + bn1 + conv2 + bn2 + flatten1 + fc1


  additional_inputs = []
  bn_list = []
  # bn_flag = False
  for name, param in simple_with_linear.named_parameters():
    # if name[:2]=='bn':
    #   additional_inputs.append(param) # dumplicate
    if name[:2]=='bn':
      # print("catch: ", name)
      # bn_flag = True
      # bn_list.insert(-1, param) # dumpicate # continue # skip bn
      bn_list.append(param)
      # print("newest bn_list: ", bn_list)
    # else:
      # bn_flag = False

    # additional_inputs.insert(-1, param)
    additional_inputs.append(param)

    # if (not bn_flag) and (len(bn_list) !=0): # False
    #   additional_inputs =additional_inputs + bn_list
    #   # print("added bn_list: ", bn_list)
    #   bn_list = []

  ### !!! add still exist bn_list if the last additional_inputs is bn- pre
  # if bn_flag and (len(bn_list) !=0):
  ### !!! add duplicated bn argus as the tile of the list
  if len(bn_list) !=0:
    # additional_inputs = additional_inputs + bn_list
    bn_list.reverse() ### !!! reverse list for bn duplicate lists
    additional_inputs = additional_inputs + bn_list
    # print("added bn_list: ", bn_list)
    bn_list = []
    # bn_flag = False

  # print("final additional_inputs: ", additional_inputs)

  # print("in mnist additional_inputs: ", additional_inputs)
  ### linear 10*20 + 20*30
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, w2_, b2_, output_value_real__, = _xla_while_loop(
  ### conv2d
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, output_value_real__, = _xla_while_loop(
  #### conv1+bn1
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, output_value_real__, = _xla_while_loop(
  ##### conv1 + bn1 + conv2
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, output_value_real__, = _xla_while_loop(
  ##### conv1 + bn1 + conv2 + bn2
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, p3, p4, p5, p6, output_value_real__, = _xla_while_loop(
  ##### conv1 + bn1 + conv2 + bn2 + fc1
  # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, p3, p4, p5, p6, p7, p8, output_value_real__, = _xla_while_loop(
  ##### conv1 + bn1 + conv2 + bn2 + fc1 + fc2 + softmax
  upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, bw1, bw11, bb1, bb11, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, output_value_real__, = _xla_while_loop(
      cond_fn, body_fn,
      (upper, lower, one_value, init_val, l_in_0, output_value), tuple(additional_inputs))
      # (upper, lower, one_value, init_val, l_in_0, output_value), ())
  print("finish newnewnew_test")
  print("torch_add_res__: run times: ", torch_add_res__)
  print("actual res: ", output_value_real__[0][0])
  expected_ = simple_with_linear(l_in_0)
  print("expected res: ", expected_[0][0])

# run test model
def test_mnist():
  torch.manual_seed(1)

  print("before test_mnist")
  newnewnew_test() # newnew_test() # new_test() # test()
  # # target fori_loop
  # for epoch in range(1, n_epochs + 1):
  #   newnewnew_test() # newnew_test() # new_test() # test()

  print("after test_mnist")

if __name__ == '__main__':
  test_mnist()

# torch.set_default_dtype(torch.float32)
# accuracy = test_mnist()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////

# import args_parse
# from torch_xla import runtime as xr

# # MODEL_OPTS = {
# #     '--ddp': {
# #         'action': 'store_true',
# #     },
# #     '--pjrt_distributed': {
# #         'action': 'store_true',
# #     },
# # }

# FLAGS = args_parse.parse_common_options(
#     datadir='/tmp/mnist-data',
#     batch_size=128,
#     momentum=0.5,
#     lr=0.01,
#     target_accuracy=98.0,
#     num_epochs=18,
#     # opts=MODEL_OPTS.items(),
# )

# import os
# import shutil
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import torch_xla
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.test.test_utils as test_utils

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch_xla.distributed.xla_backend

# import torch_xla.experimental.fori_loop
# from torch_xla.experimental.fori_loop import fori_loop


# class MNIST(nn.Module):

#   def __init__(self):
#     super(MNIST, self).__init__()
#     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#     self.bn1 = nn.BatchNorm2d(10)
#     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#     self.bn2 = nn.BatchNorm2d(20)
#     self.fc1 = nn.Linear(320, 50)
#     self.fc2 = nn.Linear(50, 10)

#   def forward(self, x):
#     x = F.relu(F.max_pool2d(self.conv1(x), 2))
#     x = self.bn1(x)
#     x = F.relu(F.max_pool2d(self.conv2(x), 2))
#     x = self.bn2(x)
#     x = torch.flatten(x, 1)
#     x = F.relu(self.fc1(x))
#     x = self.fc2(x)
#     return F.log_softmax(x, dim=1)


# def train_mnist(flags, **kwargs):
#   torch.manual_seed(1)

#   test_loader = xu.SampleGenerator(
#       data=(torch.zeros(flags.batch_size, 1, 28, 28), torch.zeros(flags.batch_size, dtype=torch.int64)),
#       sample_count=10000 // flags.batch_size // xm.xrt_world_size())

#   # Scale learning rate to num cores
#   lr = flags.lr * xm.xrt_world_size()
#   device = xm.xla_device()
#   model = MNIST().to(device)

#   # Initialization is nondeterministic with multiple threads in PjRt.
#   # Synchronize model parameters across replicas manually.
#   if xr.using_pjrt():
#     xm.broadcast_master_param(model)

#   writer = None
#   if xm.is_master_ordinal():
#     writer = test_utils.get_summary_writer(flags.logdir)
#   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
#   loss_fn = nn.NLLLoss()

#   def test_loop_fn(): # loader):
#     total_samples = 0
#     correct = 0
#     model.eval()
#     # print("loader: ", loader)
#     # print("type loader: ", type(loader))
#     # for data, target in loader:
#     for data, target in test_loader:
#       output = model(data)
#       pred = output.max(1, keepdim=True)[1]
#       correct += pred.eq(target.view_as(pred)).sum()
#       total_samples += data.size()[0]

#     accuracy = 100.0 * correct.item() / total_samples
#     accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
#     return accuracy

#   # train_device_loader = pl.MpDeviceLoader(train_loader, device)
#   test_device_loader = pl.MpDeviceLoader(test_loader, device)
#   accuracy, max_accuracy = 0.0, 0.0

#   for epoch in range(1, flags.num_epochs + 1):
#     # xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
#     # train_loop_fn(train_device_loader, epoch)
#     # xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
#     accuracy = test_loop_fn() # test_device_loader)
#     xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(epoch, test_utils.now(), accuracy))
#     max_accuracy = max(accuracy, max_accuracy)
#     # test_utils.write_to_summary(writer, epoch, dict_to_write={'Accuracy/test': accuracy}, write_xla_metrics=True)
#     # if flags.metrics_debug: xm.master_print(met.metrics_report())

#   ### fori_loop
#   # torch.set_grad_enabled(False)
#   # new_test_device_loader = pl.MpDeviceLoader(test_loader, device)
#   upper = torch.tensor([flags.num_epochs + 1], dtype=torch.int32, device=device) # flags.num_epochs + 1
#   lower = torch.tensor([1], dtype=torch.int32, device=device) # 1
#   init_val = torch.tensor([1], dtype=torch.int32, device=device)
#   # l_in_0 = torch.randn(10, device=xm.xla_device()) # test_device_loader
#   # linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
#   def body_fun():
#     res1 = torch.tensor([2], dtype=torch.int32, device=device)
#     res2 = torch.tensor([2], dtype=torch.int32, device=device)
#     res3 = res1 + res2
#     return res3
# #   def body_fun(test_device_loader):
# #     accuracy = test_loop_fn(test_device_loader)
# #     max_accuracy = max(accuracy, max_accuracy)
# #     return max_accuracy

#   upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_ = fori_loop(
#       upper, lower, body_fun, ())

#   test_utils.close_summary_writer(writer)
#   xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
#   return max_accuracy


# # def _mp_fn(index, flags):
# def main_fun(flags):
#   torch.set_default_dtype(torch.float32)
#   accuracy = train_mnist(flags)
#   if flags.tidy and os.path.isdir(flags.datadir):
#     shutil.rmtree(flags.datadir)
#   if accuracy < flags.target_accuracy:
#     print('Accuracy {} is below target {}'.format(accuracy,
#                                                   flags.target_accuracy))
#     sys.exit(21)


# if __name__ == '__main__':
# #   xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
# #   _mp_fn()
#   main_fun(FLAGS)