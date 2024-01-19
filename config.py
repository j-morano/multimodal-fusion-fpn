import argparse
import socket


"""Crop types:
Here are described the different crop types used to train the networks
using the different fusion approaches.

* 'oct': the model is trained using the OCT and the SLO resized to
  the same size as the OCT en-face projection.
* 'relative_2d': the SLO is resized to the same size as the OCT
  en-face projection, but at the feature level. This means that the
  features of the SLO are resized to the same size as the features of
  the OCT using bilinear interpolation.
* 'relative_2d_max': the same as 'relative_2d', but using max pooling.
* 'none': use the images as they are.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true')
parser.add_argument("--training-dataset", type=str, required=True)
parser.add_argument("--version", type=str, default=None)
parser.add_argument("--data-ratio", type=float, default=1.0)
parser.add_argument("--early-stopping", type=int, default=None)
parser.add_argument("--exec-test", action='store_true', help="execution test")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--val-batch-size", type=int, default=1)
parser.add_argument("--virtual-batch-size", type=int, default=1)
parser.add_argument("--compression", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=1e-1)
parser.add_argument("--fusion-modality", type=str, default=None)
parser.add_argument("--crop", type=str, default='oct')
parser.add_argument("--model", type=str, default=None, required=True)
parser.add_argument("--model-weights", type=str, default=None)
parser.add_argument("--suffix", type=str, default='')
parser.add_argument("--force-mem-cache-release", default="ReleaseMemCache")
parser.add_argument("--number-of-outputs", type=int, default=1)
parser.add_argument("--filly-annotations", type=str, default=None)
parser.add_argument("--gpus", type=int, nargs='+', default=1)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--split-indices", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--legacy-path", action='store_true')
parser.add_argument(
    "--use-complementary",
    action='store_true',
    help="Force use of complementary data",
)
parser.add_argument("--split-name", type=str, default=None)
parser.add_argument("--base-channels", type=int, default=64)
parser.add_argument(
    "--mask-variant",
    type=str,
    default='faf',
    choices=['vs_proj', 'sq_proj_dil', 'oct', 'faf'],
    help="mask variant, only for VRC vessel segmentation"
)
parser.add_argument(
    "--multiplier",
    type=int,
    default=20,
    help="Multiplier for the training dataset size."
)
parser.add_argument(
    "--rotation-augmentation",
    action='store_true',
    help="Use rotation augmentation."
)
parser.add_argument(
    "--local-server-name",
    type=str,
    default="server",
    choices=['server', 'msc_server'],
)
config, _ = parser.parse_known_args()

config.DEBUG = config.debug


# Path to save the trained models
config.models_path = f'./__server_train/{config.version}/'

# Global config
config.use_complementary = (
    'fusion' in config.model.lower()
    or '2d' in config.model.lower()
    or config.use_complementary
)

config.file_to_copy = 'run.sh'

# Model config #1
config.layers = [1, 1, 2, 4]


# Local execution
if socket.gethostname() in ['hemingway']:
    print('Running in local machine')
    # Overwrite some configs when running in local machine
    config.models_path = f'./__train/{config.version}/'
    if config.model_weights is not None:
        config.model_weights = config.model_weights.replace(
            '../',
            f'/mnt/Data/SSHFS/{config.local_server_name}/GA_SEG/'
        )
    config.batch_size = 1
    config.gpus = [0]
    config.split_indices = [0]
    config.virtual_batch_size = 1
    config.threads = 1
    config.force_mem_cache_release = "ReleaseMemCache"
    # config.base_channels = 8
    config.layers = [1, 1, 1, 1]
    config.multiplier = 1

# Model config #2
config.number_of_channels = [int(32 * 1 * 2 ** i) for i in range(0, len(config.layers))]


# Pretty print all config
print('-'*80)
print('[config]')
for k, v in config.__dict__.items():
    print(f'{k}: {v}')
print('-'*80)
