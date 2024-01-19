#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import sys
import random
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import print_net_info
from common import pl_model_wrapper, weight_init
from training_config import data_config_factory
from config import config
from models.fusion_nets import factory_classes as model_factory
import utils



data_config = data_config_factory[config.training_dataset]()


def worker_init_fn(worker_id):
    seed = torch.initial_seed() + worker_id
    np.random.seed([int(seed%0x80000000), int(seed//0x80000000)])  # type: ignore
    torch.manual_seed(seed)
    random.seed(seed)


def main(model_path, training_file_list = None, validation_file_list = None):
    print(model_path)
    print(torch.__version__)
    print(torch.backends.cudnn.version())
    print(*torch.__config__.show().split("\n"), sep="\n")
    pl.seed_everything(1234)

    torch.backends.cudnn.benchmark = True
    torch.set_anomaly_enabled(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if training_file_list is None or validation_file_list is None:
        print('The training or validation list is empty')

    print("===> Building model")
    arch = model_factory[config.model]()
    if config.model_weights is None:
        print('Random initialization')
        arch.apply(weight_init.weight_init)

    print("===> Loading datasets")
    print('Train data:', data_config.paths['oct'])

    print('Train:', training_file_list)
    print('Val:', validation_file_list)

    if isinstance(training_file_list, list):
        print('Number of training samples:', len(training_file_list))
    if isinstance(validation_file_list, list):
        print('Number of validation samples:', len(validation_file_list))

    data_transform, data_transform_val = data_config.get_transforms()

    train_data = data_config.train_data(training_file_list, data_transform)
    val_data = data_config.val_data(validation_file_list, data_transform_val)

    num_workers = config.threads
    if config.batch_size is None:
        batch_size = data_config.batch_size
    else:
        batch_size = config.batch_size
    training_data_loader = DataLoader(
        dataset=train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    evaluation_data_loader = DataLoader(
        dataset=val_data,
        num_workers=num_workers,
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=False
    )

    criterion = data_config.get_criterion()

    metrics_train = data_config.metrics_train
    metrics_val = data_config.metrics_val
    meta_metric_val = data_config.meta_metric_val

    callbacks = []

    # PL callback to store top-5 models (Dice)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='{epoch}-{'+data_config.monitor+':.4f}',
        save_top_k=5,
        monitor=data_config.monitor,  # 'Dice'
        mode=data_config.monitor_mode,
        save_weights_only=True,
    )
    callbacks.append(checkpoint_callback)

    if config.early_stopping is not None:
        early_stop_callback = EarlyStopping(
            monitor=data_config.monitor,
            min_delta=0.00,
            patience=config.early_stopping,
            verbose=True,
            mode=data_config.monitor_mode,
        )
        callbacks.append(early_stop_callback)

    optimizers = [
        torch.optim.SGD(
            arch.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        ),
    ]

    compiled_model = pl_model_wrapper.Model(
        model=arch,
        losses=criterion,
        training_metrics=metrics_train,
        metrics=metrics_val,
        metametrics=meta_metric_val,
        optim=optimizers,
        force_mem_cache_release=config.force_mem_cache_release,
        model_path=model_path,
    )

    if config.model_weights is not None:
        print('Loading pretrained model')
        checkpoint = torch.load(config.model_weights)
        try:
            state_dict = checkpoint['state_dict']
        except KeyError:
            state_dict = checkpoint
        compiled_model.load_state_dict(state_dict, strict=True)

    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        precision=32,
        devices=config.gpus,
        num_sanity_val_steps=2,
        accumulate_grad_batches=config.virtual_batch_size,
        max_epochs=config.epochs,
        sync_batchnorm=False,
        benchmark=True,
        accelerator='gpu',
        strategy='dp',
    )

    print_net_info(arch)

    if config.exec_test:
        print(arch)
        print('Testing mode enabled. Skipping training.')
        return

    print("===> Begin training")
    trainer.fit(
        compiled_model,
        train_dataloaders=training_data_loader,
        val_dataloaders=evaluation_data_loader
    )

    # Do not save model if training was interrupted
    if trainer.state.status == 'interrupted':
        print('Training interrupted')
    else:
        print("===> Saving last model")
        trainer.save_checkpoint(os.path.join(model_path, 'last.ckpt'), weights_only=True)


def train_with_split(split, idx, split_path):
    # Build model path and create dir if it does not exist
    model_path = utils.get_model_path(config, split_path, idx)

    assert isinstance(model_path, str)

    Path(model_path).mkdir(exist_ok=True, parents=True)
    print(model_path)

    if Path(join(model_path, 'last.ckpt')).exists():
        print('Model already trained. Skipping.')
        exit(0)

    # Copy execution files (for logging purposes)
    shutil.copy2(config.file_to_copy, model_path)

    # Directory to store debugging images
    Path(os.path.join(model_path, 'images')).mkdir(exist_ok=True, parents=True)

    train_ids, val_ids = split['train'], split['val']

    if config.data_ratio < 1.0:
        print('Using only', config.data_ratio*100, '% of the training data.')
        train_ids = train_ids[:int(len(train_ids)*config.data_ratio)]

    print('Number of training samples:', len(train_ids))
    print('Number of validation samples:', len(val_ids))

    main(model_path, train_ids, val_ids)


if __name__ == "__main__":

    print(sys.path)

    split_name = config.split_name
    if split_name is not None:
        split_parent = Path(data_config.paths['split']).parent
        if not split_name.endswith('.json'):
            split_name += '.json'
        split_path = split_parent / split_name
        data_config.paths['split'] = split_path
    else:
        split_path = data_config.paths['split']

    with open(split_path, 'r') as fp:
        splits = json.load(fp)

    print('Split:', Path(split_path).stem)

    if isinstance(splits, dict):
        print(
            'Only one split, ignoring split indices.'
            ' Regular training setting.'
        )
        train_with_split(splits, None, split_path)
    elif isinstance(splits, list):
        print(
            f'Multiple splits ({len(splits)}), using split indices.'
            ' Training in a cross-validation setting.'
        )
        for idx, split in enumerate(splits):
            if idx not in config.split_indices:
                continue
            print(
                'Running {} out of {} splits.'
                .format(idx, len(splits)-1)
            )
            train_with_split(split, idx, split_path)
