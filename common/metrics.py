from typing import Union, Optional

import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import hd, hd95
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, confusion_matrix,
    jaccard_score, average_precision_score, f1_score
)



class GlobalMetric:
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        raise NotImplementedError


class AUROC(GlobalMetric):
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        return roc_auc_score(gt_values, pred_values)  # type: ignore


class AUPR(GlobalMetric):
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        # Data to plot precision - recall curve
        precision, recall, _ = precision_recall_curve(gt_values, pred_values)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        return auc_precision_recall


class AP(GlobalMetric):
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        return average_precision_score(gt_values, pred_values)  # type: ignore


class F1(GlobalMetric):
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        return f1_score(gt_values, np.where(pred_values>=0.5, 1., 0.))  # type: ignore


# class for computing spec, sens and acc
class ClassMetric(GlobalMetric):
    def calculate(self, gt_values: np.ndarray, pred_values: np.ndarray) -> float:
        # Compute confusion confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            gt_values, np.where(pred_values>=0.5, 1., 0.)
        ).ravel()
        return self.get_metric(tn, fp, fn, tp)

    def get_metric(self, tn, fp, fn, tp) -> float:
        raise NotImplementedError


class Spec(ClassMetric):
    def get_metric(self, tn, fp, fn, tp) -> float:
        return tn / (tn + fp)


class Sens(ClassMetric):
    def get_metric(self, tn, fp, fn, tp) -> float:
        return tp / (tp + fn)


class Acc(ClassMetric):
    def get_metric(self, tn, fp, fn, tp) -> float:
        return (tp + tn) / (tp + tn + fp + fn)


class Metrics(object):
    def __init__(self):
        self.accumulator = []

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        return 0

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        self.accumulator.extend(result.tolist())

    def get(self):
        return np.nanmean(self.accumulator)

    def reset(self):
        self.accumulator = []


class Precision(Metrics):
    def __init__(
        self,
        output_key: Union[int,str]=0,
        target_key: Union[int,str]=0,
        slice=0
    ):
        super(Precision, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

    def calculate_batch(self, ground:dict, predict:dict):

        pred = predict[self.output_key][:,self.slice].detach()
        gr = ground[self.target_key][:,self.slice].detach()

        assert (gr.shape == pred.shape)

        N = pred.size(0)

        pred = (pred>0.5).float().view(N,-1)
        gr = (gr>0.5).float().view(N,-1)

        tp = (pred * gr).sum(dim=1).cpu().numpy()

        denominator = pred.sum(dim=1).cpu().numpy()

        p = tp / denominator
        p[denominator==0.] = 1

        return p

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        if not result is None:
            self.accumulator.extend(result.tolist())


class Recall(Metrics):
    def __init__(
        self,
        output_key: Union[int,str]=0,
        target_key: Union[int,str]=0,
        slice=0
    ):
        super(Recall, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

    def calculate_batch(self, ground:dict, predict:dict):
        pred = predict[self.output_key][:,self.slice].detach()
        gr = ground[self.target_key][:,self.slice].detach()

        assert (gr.shape == pred.shape)

        N = pred.size(0)

        pred = (pred>0.5).float().view(N,-1)
        gr = (gr>0.5).float().view(N,-1)

        tp = (pred * gr).sum(dim=1).cpu().numpy()

        denominator = gr.sum(dim=1).cpu().numpy()

        r = tp / denominator
        r[denominator==0.] = 1

        return r

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        if not result is None:
            self.accumulator.extend(result.tolist())

class IoU(Metrics):
    """Intersection over Union (Jaccard index) metric."""
    def __init__(
        self,
        output_key: Union[int,str]=0,
        target_key: Union[int,str]=0,
        slice: int=0
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        pred_slice = self.slice
        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        N = pred.size(0)

        pred = (pred[:,pred_slice]>0.5).float().view(N,-1)
        gr = (gr[:,self.slice]>0.5).float().view(N,-1)

        iou = jaccard_score(gr.cpu().numpy(), pred.cpu().numpy(), average='micro')

        return iou


class MSE(Metrics):
    def __init__(self, output_key: str, target_key: str):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:

        pred = predict[self.output_key]
        gr = ground[self.target_key]

        assert (len(gr) == len(pred))

        result = (pred - gr)**2

        return result.mean(dim=1).cpu().numpy()

    def get(self):
        return np.nanmean(self.accumulator)


class Dice(Metrics):
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        slice: int=0,
        output_threshold: float=0.5,
        target_threshold: float=0.5,
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice
        self.output_threshold = output_threshold
        self.target_threshold = target_threshold

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        pred_slice = self.slice
        try:
            assert gr[:,self.slice].shape == pred[:,self.slice].shape, (
                f"GT: {gr.shape}, Pred.: {pred.shape}"
            )
        except IndexError:
            print(f"Index: {self.slice}, GT: {gr.shape}, Pred.: {pred.shape}")

        N = pred.size(0)

        pred = (pred[:,pred_slice]>self.output_threshold).float().view(N,-1)
        gr = (gr[:,self.slice]>self.target_threshold).float().view(N,-1)

        numerator = (pred * gr).sum(dim=1).cpu().numpy()
        denominator = (pred + gr).sum(dim=1).cpu().numpy()

        r = 2 * numerator / denominator
        r[denominator==0.] = 1

        return r


class BCE(Metrics):
    """Binary Cross Entropy metric."""
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        bg_weight: float=1.0,
        slice: Optional[int]=0,
    ):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.bg_weight=bg_weight
        self.counter = 0
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        if self.slice is not None:
            assert (gr[:,self.slice].shape == pred[:,self.slice].shape)
            pred = pred[:,self.slice].view(-1)
            gr = gr[:,self.slice].view(-1)

        loss = F.binary_cross_entropy(pred, gr, reduction='mean')

        return np.array([loss.cpu().numpy()])


class L2(Metrics):
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        slice:int = 0
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape), \
            f"GT: {gr.shape}, Pred.: {pred.shape}"

        l2_distance = np.array([F.mse_loss(pred, gr, reduction='mean').cpu().numpy()])

        return l2_distance


class L1(Metrics):
    def __init__(self, output_key=0, target_key=0, slice:int = 0):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape)

        l1_distance = np.array([F.l1_loss(pred, gr, reduction='mean').cpu().numpy()])

        return l1_distance


class WeightedL1(Metrics):
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        slice: int=0,
        mask: bool=True,
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice = slice
        self.mask = mask

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape)

        l1_distance = F.l1_loss(pred, gr, reduction='none')
        l1_distance = l1_distance * ground['weight']
        if self.mask:
            l1_distance = l1_distance[ground['weight'] > 0].mean()
        else:
            l1_distance = l1_distance.mean()

        l1_distance = l1_distance.cpu().numpy()
        l1_distance = np.array([l1_distance])

        return l1_distance


class Hausdorff(Metrics):
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        slice: int=0
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice
        self.hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        pred = (pred>0.5).int().cpu().numpy()
        gr = (gr>0.5).int().cpu().numpy()

        result = []

        for n in range(pred.shape[0]):
            p = pred[n,self.slice].astype(np.uint8)
            g = gr[n,self.slice].astype(np.uint8)

            if (p.sum() == 0) or (g.sum() == 0):
                result.append(np.nan)
                continue
            else:

                try:
                    spacing = ground['spacing'][n].cpu().numpy()
                except KeyError:
                    affine = ground['affine'][n].cpu().numpy()
                    spacing = np.sqrt((affine ** 2).sum(axis=0))[:len(p.shape)] #type:ignore
                try:
                    r = hd(p[:,0],g[:,0],voxelspacing=spacing[[0,2]])
                    result.append(r)
                except RuntimeError as E:
                    print("Hausdorff:RuntimeError: "+ str(E))
                    pass

        return np.array(result)


class Hausdorff95(Metrics):
    def __init__(
        self,
        output_key: Union[str, int]=0,
        target_key: Union[str, int]=0,
        slice: int=0
    ):
        super().__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr[:,self.slice].shape == pred[:,self.slice].shape)

        pred = (pred>0.5).int().cpu().numpy()
        gr = (gr>0.5).int().cpu().numpy()

        result = []

        for n in range(pred.shape[0]):
            p = pred[n,self.slice].astype(np.uint8)
            g = gr[n,self.slice].astype(np.uint8)

            if (p.sum() == 0) or (g.sum() == 0):
                result.append(np.nan)
                continue
            else:

                try:
                    spacing = ground['spacing'][n].cpu().numpy()
                except KeyError:
                    affine = ground['affine'][n].cpu().numpy()
                    spacing = np.sqrt((affine ** 2).sum(axis=0))[:len(p.shape)] #type:ignore
                try:
                    r = hd95(p[:,0],g[:,0],voxelspacing=spacing[[0,2]],connectivity=3)
                    result.append(r)
                except RuntimeError as E:
                    print("Hausdorff95:RuntimeError: "+ str(E))
                    pass

        return np.array(result)
