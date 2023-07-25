import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bc")

# set config attributes here that you would like to update
config.experiment.name = "bc_rnn_example"
config.train.data = "./data/test_base_wrapper.hdf5"
config.train.output_dir = "./output_dir/"
config.train.batch_size = 256
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)