from os.path import join
from typing import Optional, Union

import torch.utils.data as data
from common.mytransforms import Compose

from config import config


class AbstractDataset(data.Dataset):
    """Abstract class for datasets.
    This class is used to define the common interface that all datasets
    should follow. It also implements some common functionality that is
    shared across all datasets.
    """

    def __init__(self):
        super().__init__()
        # Attributes
        self.visits: dict
        self.path: str
        self.real_length: int
        self.path: str
        self.multiplier: int
        self.patches_from_single_image: int
        self.transforms: Optional[Compose]
        self.mask_variant: Optional[str]
        self.get_spacing: bool
        self.patients: Optional[Union[dict, list]]
        self.patches_from_current_image: int
        self.record: dict


    def _load(self, _index):
        raise NotImplementedError


    def _make_abstract_dataset(self) -> list:
        # When using directly the scan ids, self.patients is a dict with
        #   the format {'ids': list_of_ids}. Otherwise, it is a list of
        #   patient ids.
        if isinstance(self.patients, dict):
            self.dataset = self._make_dataset_ids(ids=self.patients['ids'])
        elif isinstance(self.patients, list):
            self.dataset = self._make_dataset(patients=self.patients)
        else:
            raise ValueError
        return self.dataset


    def _make_dataset_ids(self, ids: list) -> list:
        dataset = []

        for k in self.visits:
            for visit in self.visits[k]:
                if visit['FileSetId'] in ids:
                    record = {}
                    record['path'] = join(self.path, visit['FileSetId'])
                    record['FileSetId'] = visit['FileSetId']
                    record['DayInStudy'] = visit['DayInStudy']
                    record['VRCPatId'] = k
                    record['Position'] = visit['Position']
                    record['slo_path'] = join(
                        self.path,
                        k+'_'+visit['Position'],
                        str(visit['DayInStudy'])
                    )

                    dataset.append(record)

        return dataset

    def _make_dataset(self, patients: Union[dict, list]) -> list:
        dataset = []

        for k in patients:
            for visit in self.visits[k]:
                record = {}
                record['path'] = join(self.path, visit['FileSetId'])
                record['FileSetId'] = visit['FileSetId']
                record['DayInStudy'] = visit['DayInStudy']
                record['VRCPatId'] = k
                record['Position'] = visit['Position']
                record['slo_path'] = join(
                    self.path,
                    k+'_'+visit['Position'],
                    str(visit['DayInStudy'])
                )

                dataset.append(record)

        return dataset

    def __getitem__(self, index):
        index = index % self.real_length

        if self.patches_from_current_image >= self.patches_from_single_image:
            self._load(index)
            self.patches_from_current_image = 0

        self.patches_from_current_image += 1

        record = self.record.copy()

        if self.transforms is not None:
            record = self.transforms(record)

        if config.DEBUG:
            for k, v in record.items():
                try:
                    print('__getitem__', k, v.shape)
                except:
                    pass

        return record

    def __len__(self):
        return int(self.multiplier * self.real_length)
