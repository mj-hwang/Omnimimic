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
config.train.data = "./data/tiago_nav_pick_25nav_20pick.hdf5"

config.train.output_dir = "./testingg/"
config.train.batch_size = 256
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# config.observation.low_dim.do_not_lock_keys()

# config.observation.modalities = ["low_dim", "rgb", "depth", "scan"]
config.observation.modalities.obs.low_dim = ["proprio"]
config.observation.modalities.obs.rgb = ["rgb"]
# config.observation.modalities.obs.depth = ["depth"]
# config.observation.modalities.obs.scan = ["scan"]


# =============== Low Dim default encoder (no encoder) ===============
config.observation.encoder.low_dim.core_class = None
config.observation.encoder.low_dim.core_kwargs = Config()                 
config.observation.encoder.low_dim.core_kwargs.do_not_lock_keys()

# =============== RGB default encoder (ResNet backbone + linear layer output) ===============
config.observation.encoder.rgb.core_class = "VisualCore"     
config.observation.encoder.rgb.core_kwargs = Config()
config.observation.encoder.rgb.core_kwargs.input_shape = [3, 128, 128]
config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
config.observation.encoder.rgb.core_kwargs.backbone_kwargs = Config()
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
config.observation.encoder.rgb.core_kwargs.pool_kwargs = Config()
config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
# {
#     "input_shape": [3, 128, 128],
#     "backbone_class": "ResNet18Conv",  # use ResNet18 as the visualcore backbone
#     "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
#     "pool_class": "SpatialSoftmax",  # use spatial softmax to regularize the model output
#     "pool_kwargs": {"num_kp": 32}
# }
# config.observation.encoder.rgb.core_kwargs.do_not_lock_keys()

# # =============== Depth default encoder (same as rgb) ===============
# config.observation.encoder.depth.core_class = "VisualCore"     
# config.observation.encoder.depth.core_kwargs = Config()
# config.observation.encoder.depth.core_kwargs.do_not_lock_keys()

# =============== Scan default encoder (Conv1d backbone + linear layer output) ===============
# config.observation.encoder.scan.core_class = "ScanCore"               
# config.observation.encoder.scan.core_kwargs = Config()
# config.observation.encoder.scan.core_kwargs.do_not_lock_keys()

config.experiment.env = None
config.experiment.rollout.enabled = False

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

breakpoint()

# launch training run
train(config, device=device)
