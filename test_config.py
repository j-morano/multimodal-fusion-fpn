from training_config import (
    DefaultConfig,
    HRFConfig,
    HRFFusionConfig,
    VRCVConfig,
    HRFFusionCompOnlyConfig,
    VRCVLR2CompOnlyConfig,
)
from common import metrics

from utils import get_factory_adder



add_class, opt_factory = get_factory_adder()


class OptDefaults(DefaultConfig):
    results_path = './__test/'
    models_path = './__train/'

    device = 'cuda'
    which_model = 'top-k'
    save_data = True

    metrics_val = {
        'Dice': metrics.Dice(output_key='prediction', target_key='mask', slice=0),
        'BCE': metrics.BCE(output_key='prediction', target_key='mask', slice=0),
        'Precision': metrics.Precision(output_key='prediction', target_key='mask'),
        'Recall': metrics.Recall(output_key='prediction', target_key='mask'),
        'Hausdorff': metrics.Hausdorff(output_key='prediction', target_key='mask', slice=0),
        'Hausdorff95': metrics.Hausdorff95(output_key='prediction', target_key='mask', slice=0),
    }

    global_metrics = {}

    def __str__(self) -> str:
        """Pretty-prints all attributes of the class."""
        attrs = dir(self)
        # Get values of all attributes
        values = [getattr(self, attr) for attr in attrs]
        # Create a dictionary of all attributes and their values
        attr_dict = dict(zip(attrs, values))
        # Add attributes from __dict__ to attr_dict
        attr_dict.update(self.__dict__)
        string = f'# {self.__class__.__name__}:\n'
        for key, value in attr_dict.items():
            if not key.startswith('__'):
                string += f'  * {key}: {value}\n'
        return string


@add_class('hrf')
class HRFOpt(HRFConfig, OptDefaults):
    ...


@add_class('hrf_fusion')
class HRFFusionOpt(HRFFusionConfig, OptDefaults):
    ...


@add_class('vrc')
class VRCOpt(VRCVConfig, OptDefaults):
    global_metrics = {
        'AUROC': metrics.AUROC(),
        'AUPR': metrics.AUPR(),
        'Sens': metrics.Sens(),
        'Spec': metrics.Spec(),
        'Acc': metrics.Acc(),
        'AP': metrics.AP(),
        'F1': metrics.F1(),
    }

    metrics_val = {
        'Dice': metrics.Dice(output_key='prediction', target_key='mask', slice=0),
        'Precision': metrics.Precision(output_key='prediction', target_key='mask'),
        'Recall': metrics.Recall(output_key='prediction', target_key='mask'),
        'IoU': metrics.IoU(output_key='prediction', target_key='mask', slice=0),
    }


@add_class('hrf_fusion_comp_only')
class HRFFusionCompOnlyOpt(HRFFusionCompOnlyConfig, OptDefaults):
    ...


@add_class('vrc_lr2_comp_only')
class VRCLR2CompOnlyOpt(VRCVLR2CompOnlyConfig, OptDefaults):
    ...
