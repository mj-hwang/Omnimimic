import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.config.config import Config
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bc")

# set config attributes here that you would like to update
config.experiment.name = "bc_rnn_example"
# config.train.data = "./data/test_base_wrapper.hdf5"
config.train.data = "./data/tiago_nav_20traj.hdf5"

config.train.output_dir = "./output_test_nav/"
config.train.batch_size = 256
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# config.observation.low_dim.do_not_lock_keys()

# config.observation.modalities = ["low_dim", "rgb", "depth", "scan"]
config.observation.modalities.obs.low_dim = ["proprio"]

# =============== Low Dim default encoder (no encoder) ===============
config.observation.encoder.low_dim.core_class = None
config.observation.encoder.low_dim.core_kwargs = Config()                 
config.observation.encoder.low_dim.core_kwargs.do_not_lock_keys()

config.experiment.env = None
config.experiment.rollout.enabled = False

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)

