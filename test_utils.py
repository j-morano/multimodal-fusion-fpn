from pathlib import Path
import json
import os
from os.path import join
from typing import Optional

import pandas as pd
from skimage.transform import resize
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
import numpy as np
from skimage import io

import utils
from noise import noise_classes



def average_outputs(outputs, dtype):
    if isinstance(outputs,list) and dtype==dict:
        keys = outputs[0].keys()
        return {
            key: average_outputs(
                [d[key] for d in outputs],
                dtype=type(outputs[0][key])
            )
            for key
            in keys
        }

    elif isinstance(outputs,list) and dtype==str:
        return outputs[0]
    elif isinstance(outputs, list) and dtype == torch.Tensor:
        return sum(outputs)/len(outputs)
    else:
        assert()


def compute_global_metrics(global_metrics, all_outputs, output_path):
    print('\nGlobal metrics:')
    gm_results = {}
    for gm in global_metrics:
        m_value = global_metrics[gm].calculate(all_outputs['gt'], all_outputs['pred'])
        print(f'{gm}: {m_value}')
        gm_results[gm] = m_value
    with open(join(output_path, 'global_metrics.json'), 'w') as fp:
        json.dump(gm_results, fp, indent=4)


def get_final_results(
        output_path,
        metrics_val,
        df_results,
        df_mean_eyes,
        df_mean_patients,
        show_plots=False
):
    mean_results = {}
    for k in metrics_val.keys():
        mean_results[k] = {
            'mean': df_results[k].mean(),
            'std': df_results[k].std(),
            'median': df_results[k].mean(),
            'q25': df_results[k].quantile(0.25),
            'q75': df_results[k].quantile(0.75),
        }
        print(
            'Mean {}: {} std {} Median {} Q25 {} Q75 {}'.format(
                k,
                df_results[k].mean(),
                df_results[k].std(),
                df_results[k].median(),
                df_results[k].quantile(0.25),
                df_results[k].quantile(0.75)
            )
        )
        print(
            'Mean eye {}: {} std {} Median {} Q25 {} Q75 {}'.format(
                k,
                df_mean_eyes[k].mean(),
                df_mean_eyes[k].std(),
                df_mean_eyes[k].median(),
                df_mean_eyes[k].quantile(0.25),
                df_mean_eyes[k].quantile(0.75)
            )
        )

        print(
            'Mean patient {}: {} std {} Median {} Q25 {} Q75 {}'.format(
                k,
                df_mean_patients[k].mean(),
                df_mean_patients[k].std(),
                df_mean_patients[k].median(),
                df_mean_patients[k].quantile(0.25),
                df_mean_patients[k].quantile(0.75)
            )
        )

        if show_plots:
            sns.displot(df_results[k].dropna(), label=k)
            sns.displot(df_mean_eyes[k].dropna(), label=k + ' grouped by eye')
            sns.displot(df_mean_patients[k].dropna(), label=k + ' grouped by patient')
            plt.legend()
            plt.title(k)
            plt.show()

    mean_results_file = os.path.join(output_path, 'mean_results.json')
    with open(mean_results_file, 'w') as fp:
        json.dump(mean_results, fp, indent=4)


def get_final_results_only(
        output_path,
        metrics_val,
        df_results,
        show_plots=False
):
    mean_results = {}
    for k in metrics_val.keys():
        mean_results[k] = {
            'mean': df_results[k].mean(),
            'std': df_results[k].std(),
            'median': df_results[k].mean(),
            'q25': df_results[k].quantile(0.25),
            'q75': df_results[k].quantile(0.75),
        }
        print(
            'Mean {}: {} std {} Median {} Q25 {} Q75 {}'.format(
                k,
                df_results[k].mean(),
                df_results[k].std(),
                df_results[k].median(),
                df_results[k].quantile(0.25),
                df_results[k].quantile(0.75)
            )
        )

        if show_plots:
            sns.displot(df_results[k].dropna(), label=k)
            plt.legend()
            plt.title(k)
            plt.show()

    mean_results_file = os.path.join(output_path, 'mean_results.json')
    with open(mean_results_file, 'w') as fp:
        json.dump(mean_results, fp, indent=4)


def compute_metrics(
    all_outputs, output, batch, metrics_val, results, results_dict,
    output_path, save_data: bool=True,
):
    metrics_row = {}

    # if opt.global_metrics:
        # [0]: do it only for the first mask
    # NOTE: Always get all_outputs
    output_np = output['prediction'].cpu().numpy()
    mask_np = batch['mask'].cpu().numpy()
    # pred_shape_channels = output_np.shape[1]
    all_outputs['pred'] = np.concatenate(
        # (all_outputs['pred'], output_np[0].flatten())
        (all_outputs['pred'], output_np[0,0].flatten())
    )
    all_outputs['gt'] = np.concatenate(
        # (all_outputs['gt'], mask_np[0,:pred_shape_channels].flatten())
        (all_outputs['gt'], mask_np[0,0].flatten())
    )

    copy_list = ['VRCPatId', 'FileSetId']

    if isinstance(batch['FileSetId'], torch.Tensor):
        identifier = batch['FileSetId'].item()
    else:
        identifier = batch['FileSetId'][0]

    for c in copy_list:
        if isinstance(batch[c], torch.Tensor):
            metrics_row[c] = batch[c].item()
        else:
            metrics_row[c] = batch[c][0]

    if 'mask' in batch:
        for m, v in metrics_val.items():
            metrics_row[m] = v.calculate_batch(batch, output).item()

    # Confirm that each image is only evaluated once
    if identifier in list(results_dict):
        raise ValueError('Identifier already in results_dict')
    try:
        results_dict[identifier] = metrics_row['Dice']
    except KeyError:
        results_dict[identifier] = metrics_row['WeightedL1']

    assert 'mask' in batch and 'prediction' in output

    mask_crop = mask_np[0,0]
    output_crop = output_np[0,0]

    if 'spacing' in batch:
        spacing = batch['spacing'][0].cpu().numpy()

        metrics_row['Area'] = (output_crop > 0.5).sum()*spacing[0]*spacing[2]
        if 'mask' in batch:
            metrics_row['Area_manual'] = (mask_crop > 0.5).sum() *spacing[0]*spacing[2]
            metrics_row['Area_diff'] = metrics_row['Area'] - metrics_row['Area_manual']

    if not save_data:
        results.append(metrics_row)
        print(metrics_row)
        return

    sample_output_path = os.path.join(output_path, identifier)
    os.makedirs(sample_output_path, exist_ok=True)

    if 'out_features' in output:
        # Average feature maps from 0 to half and from half to end, so
        #   that we obtain 2 2D maps.
        out_features = output['out_features'][0].cpu().detach().numpy()
        out_features_0 = out_features[:out_features.shape[0]//2, :, 0, :]
        out_features_1 = out_features[out_features.shape[0]//2:, :, 0, :]
        out_features_0 = np.mean(out_features_0, axis=0)
        out_features_1 = np.mean(out_features_1, axis=0)
        # Create a 2D image with the 2 feature maps
        out_features = np.concatenate(
            (out_features_0, out_features_1), axis=1
        )  # type: np.ndarray
        # Save the image
        out_features = resize(
            out_features,
            (256, 256*2),
            order=1,
            preserve_range=True,
            anti_aliasing=False
        )
        try:
            io.imsave(
                os.path.join(sample_output_path, 'features.png'),
                out_features
            )
        except ValueError:
            print('Error saving features')
            print(out_features.shape)

    with open (join(sample_output_path, 'info.json'), 'w') as fp:
        json.dump(metrics_row, fp, indent=4)

    io.imsave(
        os.path.join(sample_output_path, 'test.png'),
        ((output_crop[:, 0, :] > 0.5) * 255).astype(np.uint8)
    )
    io.imsave(
        os.path.join(sample_output_path, 'test_soft.png'),
        (output_crop[:, 0, :] * 255).astype(np.uint8)
    )
    if mask_crop is not None:
        # Check if image exists, if not, save it
        mask_path = os.path.join(sample_output_path, 'mask.png')
        if not os.path.exists(mask_path):
            io.imsave(
                mask_path,
                (mask_crop[:, 0, :] * 255).astype(np.uint8)
            )

    del output, batch
    results.append(metrics_row)
    print(metrics_row)


def run_single_evaluation_instance(
    opt,
    all_outputs,
    val_ids,
    data_transform_val,
    model,
    metrics_val,
    results,
    results_dict,
    output_path,
):
    evaluation_data_loader = create_val_dataloader(opt, val_ids, data_transform_val)
    model.eval()
    model.validation = join(output_path, '__images')
    Path(model.validation).mkdir(parents=False, exist_ok=True)

    with torch.no_grad():
        for batch in evaluation_data_loader:
            batch = utils.array_to_cuda(batch, device=opt.device)
            output = model(batch)

            compute_metrics(
                all_outputs,
                output,
                batch,
                metrics_val,
                results,
                results_dict,
                output_path,
                opt.save_data
            )


def create_val_dataloader(opt, val_ids, data_transform_val) -> DataLoader:
    val_data = opt.val_data(val_ids, data_transform_val=data_transform_val)
    evaluation_data_loader = DataLoader(
        dataset=val_data,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    return evaluation_data_loader


def run_evaluation_instance(
    opt,
    all_outputs,
    val_ids,
    data_transform_val,
    models,
    metrics_val,
    results,
    results_dict,
    output_path,
    noise: Optional[str]=None,
    debug_images: bool=False,
):
    evaluation_data_loader = create_val_dataloader(opt, val_ids, data_transform_val)
    for n in models:
        models[n]['model'].eval()
        if debug_images:
            models[n]['model'].validation = join(output_path, '__images', f'model_{n}')
            Path(models[n]['model'].validation).mkdir(parents=False, exist_ok=True)
        else:
            Path(join(output_path, 'images')).mkdir(parents=False, exist_ok=True)

    with torch.no_grad():
        for batch in evaluation_data_loader:
            # batch type: Dict[str, torch.Tensor]
            batch = utils.array_to_cuda(batch, device=opt.device)

            # Format: <modality>-<noise_type>-<noise_level>
            if noise is not None and isinstance(batch, dict):
                split = noise.split('-')
                modality_to_noise = split[0]
                noise_type = split[1]
                noise_level = float(split[2])
                batch[modality_to_noise] = noise_classes[noise_type](
                    noise_level
                )(batch[modality_to_noise])

            outputs = []
            for n, v in models.items():
                # print(batch['image'].shape, batch['slo'].shape)
                output = v['model'](batch)
                outputs.append(output)

            output = average_outputs(outputs,dict)
            compute_metrics(
                all_outputs,
                output,
                batch,
                metrics_val,
                results,
                results_dict,
                output_path,
                opt.save_data
            )


def get_mean_results(opt, all_outputs, results, results_dict, output_path):
    df_results = pd.DataFrame(results)
    results_file = os.path.join(output_path, 'test_output.csv')
    df_results.to_csv(results_file)
    with open(os.path.join(output_path, 'results_dict.json'), 'w') as fp:
        json.dump(results_dict, fp, indent=4)

    df_results = pd.read_csv(results_file, index_col=0)
    get_final_results_only(output_path, opt.metrics_val, df_results)
    compute_global_metrics(opt.global_metrics, all_outputs, output_path)
