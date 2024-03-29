from os.path import join
import json
from pathlib import Path
import os
import glob
import random

import numpy as np
import torch

from config import config, parser
from common import pl_model_wrapper
from models.fusion_nets import factory_classes as model_factory
import utils
from test_config import opt_factory
from training_config import data_config_factory
from test_utils import run_evaluation_instance, get_mean_results



parser.add_argument(
    "--noise",
    type=str,
    required=False,
    default=None,
    help=(
        "An argument to specify the kind of noise applied to each"
        " image. If not specified, no noise will be applied.\n"
        "Format: <modality>-<noise_type>"
    )
)
parser.add_argument("--noise-level", type=float, required=False, default=None)
parser.add_argument(
    "--debug-images",
    action="store_true",
    help=(
        "If true, the debug images generated by the models will be saved."
    )
)
parser.add_argument(
    "--test-dataset",
    type=str,
    required=False,
    default=None,
    help=(
        "Name of the dataset to use for testing, as defined in the test"
        " configuration file. If not specified, the training dataset"
        " name will be used."
    )
)
parser.add_argument(
    "--eval-split",
    type=str,
    required=False,
    default=None,
    help=(
        "Filename (without extension) of the split to use for evaluation."
        " If not specified, the default split will be used. I.e., the one"
        " used for training."
    )
)
parser.add_argument(
    "--save-all-outputs",
    action="store_true",
    help=(
        "If true, all the outputs of the models will be saved."
    )
)
parser.add_argument(
    "--force-repeat",
    action="store_true",
    help=(
        "If true, the evaluation will be repeated even if the results"
        "  file lready exists."
    )
)
parser.add_argument(
    "--dont-save",
    action="store_true",
    help=(
        "If true, the results will not be saved."
    )
)
parser.add_argument(
    "--eval-mask-variant",
    type=str,
    required=False,
    default=None,
)
parser.add_argument("--repetition", type=int, default=-1)
args = parser.parse_args()


if args.test_dataset is None:
    args.test_dataset = args.training_dataset

opt = opt_factory[args.test_dataset]()


if args.noise is not None:
    assert args.noise_level is not None
    args.noise = f'{args.noise}-{args.noise_level}'

noise_dir = args.noise if args.noise is not None else 'no-noise'


# Fix all seeds
seed = 1234 + args.repetition
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # type: ignore
random.seed(seed)


opt.results_path = './__test/'


if args.eval_split is None:
    if args.split_name is not None:
        args.eval_split = args.split_name
        eval_split_fn = join(opt.paths['oct'], args.eval_split)
    else:
        args.eval_split = opt.paths['split']
        eval_split_fn = args.eval_split
else:
    eval_split_fn = join(opt.paths['oct'], args.eval_split)

if not eval_split_fn.endswith('.json'):
    eval_split_fn += '.json'

eval_split_name = Path(eval_split_fn).stem
# dataset_full_name = Path(args.eval_split).parent.name

print('Using split:', eval_split_fn)
with open(eval_split_fn, 'r') as f:
    eval_split = json.load(f)


if args.dont_save:
    opt.save_data = False

if args.eval_mask_variant is not None:
    opt.mask_variant = args.eval_mask_variant


print(opt)


all_outputs = {
    'pred': np.array([]),
    'gt': np.array([]),
}


def main():
    if config.training_dataset != args.test_dataset:
        test_name = f'{config.training_dataset}-to-{args.test_dataset}'
    else:
        test_name = config.training_dataset

    # Transformations are obtained from the test config, which itself
    #   gets them from the training config.
    data_transform_val = opt.get_val_transforms()

    # No need to instantiate data_config, since we only need the paths,
    #   which is a static attribute.
    if args.split_name is not None:
        training_split = join(opt.paths['oct'], args.split_name)
    else:
        training_split = data_config_factory[config.training_dataset].paths['split']

    model_paths = []
    # utils.get_model_path is the same function used in train.py.
    #   In this way, we ensure that we are using the same model.
    current_model_path, training_split_name = utils.get_model_path(config, training_split, None, True)

    print('Current model path:', current_model_path)
    assert os.path.exists(current_model_path), current_model_path

    output_path = opt.results_path
    if args.noise is not None:
        output_path = join(output_path, noise_dir)
    if args.repetition >= 0:
        output_path = join(output_path, f'rep_{args.repetition}')
    output_path = os.path.join(
        output_path,
        # current_model_path except the parent directory
        current_model_path
            .split('__train/')[1]
            .replace(config.training_dataset, test_name),
    )
    if training_split_name != eval_split_name:
        if args.eval_mask_variant is not None:
            output_path = output_path.replace(
                training_split_name,
                "{}-to-{}--{}".format(
                    training_split_name,
                    eval_split_name,
                    args.eval_mask_variant
                ),
            )
        else:
            output_path = output_path.replace(
                training_split_name,
                "{}-to-{}".format(training_split_name, eval_split_name),
            )
    print('\n>>> Output path: {}\n'.format(output_path))

    if args.save_all_outputs:
        if os.path.exists(join(output_path, 'all_outputs.npz')) and not args.force_repeat:
            print('All outputs file already exists. Skipping.')
            exit(0)
    elif os.path.exists(join(output_path, 'mean_results.json')) and not args.force_repeat:
        print('Results file already exists. Skipping.')
        exit(0)

    Path(join(output_path, '__images')).mkdir(parents=True, exist_ok=True)

    # Get the paths of the top-k models (in terms of validation loss)
    #   saved during training.
    model_paths = glob.glob(join(current_model_path, 'epoch=*.ckpt'))
    print('Model paths ({}): {}'.format(len(model_paths), model_paths))
    assert len(model_paths) == 5, model_paths
    # last_model = glob.glob(join(current_model_path, 'last.ckpt'))
    # assert len(last_model) == 1, last_model

    # Then, load all the models
    models = {}
    for n, model_path in enumerate(model_paths):
        arch = model_factory[config.model]()

        # NOTE: remember to set the 'validation' argument later, to
        #   save test images in a convenient format
        # E.g. join(output_path, '__images')
        # Needed params for compiling the model
        params = {
            'losses':None,
            'metrics':None,
            'metametrics':None,
            'optim':None,
            'training_metrics':None,
            'validation': None,
            'model_path': output_path,
        }
        compiled_model = pl_model_wrapper.Model(model=arch, **params)
        path_weights = model_path
        print(f'Loading weights from {path_weights}')
        checkpoint = torch.load(
            path_weights, map_location=torch.device(opt.device)
        )
        # Replace all 'resensenet' with 'resnet' in the state dict
        #   (this is a hack to load the weights of the old models)
        checkpoint['state_dict'] = {
            k.replace('resensenet', 'resensnet'): v
            for k, v in checkpoint['state_dict'].items()
        }
        compiled_model.load_state_dict(checkpoint['state_dict'], strict=True)
        compiled_model = compiled_model.eval().to(opt.device)
        models[n] = {
            'model': compiled_model,
            'path_weights': path_weights,
            'path': model_path,
        }

    for k, v in models.items():
        print('-' * 80)
        print(f'Model {k}: {v["model"].__class__}, {v["path_weights"]}')
        print(f'Path: {v["path"]}')


    results = []
    results_dict = dict()

    if isinstance(eval_split, list):
        # Do this when directly passing a list of scan ids
        val_ids = {
            'ids': eval_split,
        }
    elif isinstance(eval_split, dict):
        # Otherwise, assume that it is a split file, so get the patient
        #   ids of the test set.
        val_ids = eval_split['test']
    else:
        raise ValueError('Unknown split data type')

    if config.exec_test:
        print('Skipping. exec_test is True.')
        exit(0)

    run_evaluation_instance(
        opt,
        all_outputs,
        val_ids,
        data_transform_val,
        models,
        opt.metrics_val,
        results,
        results_dict,
        output_path,
        noise=args.noise,
        debug_images=args.debug_images,
    )
    get_mean_results(
        opt,
        all_outputs,
        results,
        results_dict,
        output_path
    )

    if args.save_all_outputs:
        assert all_outputs['pred'].shape == all_outputs['gt'].shape
        # Save in compressed format.
        np.savez_compressed(
            join(output_path, 'all_outputs.npz'),
            pred=all_outputs['pred'],
            gt=all_outputs['gt'],
        )


if __name__ == '__main__':
    main()
