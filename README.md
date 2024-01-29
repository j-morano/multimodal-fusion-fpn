# 3D->2D segmentation using multimodal 3D+2D image fusion

This is the official repository of the paper ["Deep Multimodal Fusion of Data with Heterogeneous Dimensionality via Projective Networks"](https://doi.org/10.1109/JBHI.2024.3352970), published in _IEEE Journal of Biomedical and Health Informatics_ on January 2024.

For questions about the code or the paper, please contact the first author, José Morano: <jose.moranosanchez@meduniwien.ac.at>, or open an issue in this repository (preferred).

## Framework

![image](https://github.com/j-morano/multimodal-fusion-fpn/assets/48717183/44282cbe-22f7-49ed-9e32-7c9765d3b3f5)
Our proposed framework defines a novel fusion approach that extracts and projects the features of all modalities into the feature space of the modality with the lowest dimensionality (_n_), so that they can be employed for localization tasks in the common (_n_-dimensional, _n_D) subspace.


## Setting up the environment

Install `pyenv`.
```sh
curl https://pyenv.run | bash
```

Install `clang`. _E.g._:
```sh
sudo dnf install clang
```

Install Python version 3.6.8.
```sh
CC=clang pyenv install -v 3.6.8
```

Create and activate Python environment.
```sh
~/.pyenv/versions/3.6.8/bin/python3 -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Update `pip`.

```sh
pip install --upgrade pip
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```

## Running the code

See the options in `config.py` and sample commands in `run.sh`.

The main script for running the training is `train.py`, while `validate_ensemble.py` is used for testing.

The dataloaders expect the data to be (approximately) in the following format.
The `id0` is the image ID.
Please check the dataloaders for more details and modify them as needed.

```
dataset_dir/
    id0/
        preprocessed_images/
            bscan_size.mask.id0.png
        slo.id0.png
        faf.id0.png
        bscan.id0.npy
        spacing.id0.npy
```

Spacing is needed for the computation of the Hausdorff distance.



## Other information

Level5 versions of the architectures differ from the other versions in that the fusion is done at the 5 levels, not only the first 4.


## Citation

If you use this code, please cite the following paper:

```
@ARTICLE{morano2024deep,
  author={Morano, José and Aresta, Guilherme and Grechenig, Christoph and Schmidt-Erfurth, Ursula and Bogunović, Hrvoje},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={Deep Multimodal Fusion of Data with Heterogeneous Dimensionality via Projective Networks},
  year={2024},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2024.3352970}
}
