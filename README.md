# IMUMapNet 

## Purpose
This project is based on [MapNet](https://github.com/NVlabs/geomapnet) and modifies the code implementation to incorporate IMU sensors as an additional supervision signal. This is based on a Python 3 fork from [here](https://github.com/mcimpoi/geomapnet). The original readme can be found [here](README_og.md).

## Setup

MapNet uses a Conda environment that makes it easy to install all dependencies.

1. Create the `mapnet` Conda environment: `conda env create -f environment_imu.yml`.
2. Activate the environment: `conda activate mapnet_36`.

## Data
For this project, two datasets captured using [this code](https://github.com/sytan98/Dataset-Collection-for-Pose-and-IMU) were used in the experiments:
- Synthetic dataset from AirSim: building
- Real-world datasets: campus

The dataset loader ```dataset_loaders/airsim.py``` is used to load for both datasets as they are in the same data format.
Datasets can be found [here](https://drive.google.com/drive/folders/1DlQqAXiIvlrzFM5Ruxc_9OD_6XnZcGpf?usp=sharing)

## Running the code
Google Colab was used to run the experiments. It runs experiments by using ```scripts/train.py``` and ```scripts/eval.py```. 
- ```imu_experiments\IMUMapNet_synthetic_experiments.ipynb```: Experiments on synthetic dataset.
- ```imu_experiments\IMUMapNet_real_world_experiments.ipynb```: Experiments on real-world dataset.

Tensorboard was also used instead of visdom to visualise training and testing graphs.

### Train
Experiments involve:
- Train using varying noise levels:
This is varied as an argument when you run the python script ```--noisy_training None/v1/v2```
- Train weighted loss function model with varying alpha
This is done using ```--imu_mode Average```. IMU weight alpha can be varied in ```configs/mapnet.ini``` as the imu_loss_weight parameter
- Train averaging output model with simple average and with SLERP
This is done using ```--imu_mode Separate```. Averaging method is varied using ```--average_method Simple/Interpolate```
- Vary skip size 
This is varied in the ```configs/mapnet.ini``` as the skip parameter

The executable script is `scripts/train.py`. Please go to the `scripts` folder to run these commands. For example:

- Baseline MapNet on `building`:

```	
$ python train.py \
--dataset AirSim --scene building --data_dir '/content/drive/MyDrive/Colab Notebooks/FYP/datasets' \
--config_file configs/mapnet.ini --model mapnet --device 0 \
--imu_mode None \
--noisy_training None 
```

- Weighted loss function MapNet on `campus`:

```
$ python train.py \
--dataset AirSim --scene campus --data_dir '/content/drive/MyDrive/Colab Notebooks/FYP/datasets' \
--config_file configs/mapnet.ini --model mapnet --device 0 \
--imu_mode Average \
--noisy_training None \
--suffix _imu_weight_0_5
```

### Evaluation
The trained models for all experiments presented in the paper can be downloaded [here](https://drive.google.com/drive/folders/1ZPJqqEizp0vR1rkjzIDAdyl4sRTtaArE?usp=sharing).
The inference script is `scripts/eval.py`. Here are some examples, assuming the models are downloaded in `scripts/logs`. Please go to the `scripts` folder to run the commands.

- Baseline MapNet on `building`:
```
$ python eval.py \
--dataset AirSim --scene building --data_dir '/content/drive/MyDrive/Colab Notebooks/FYP/datasets' \
--config_file configs/mapnet.ini --model mapnet --output_dir results/ \
--weights logs/AirSim_building_mapnet_mapnet_imu_None_noisy_None/epoch_100.pth.tar \
--val --imu_mode None   \
--plot_3d
```

- Averaging output MapNet on `building`:
```
$ python eval.py \
--dataset AirSim --scene building --data_dir '/content/drive/MyDrive/Colab Notebooks/FYP/datasets' \
--config_file configs/mapnet.ini --model mapnet --output_dir results/ \
--weights logs/AirSim_building_mapnet_mapnet_imu_Separate_noisy_v2/epoch_100.pth.tar \
--val --imu_mode Separate \
--average_method Simple
--plot_3d 
```

### Other Tools 
#### dataset_utils.py
This script can be used to calculate mean and stdev pixel statistics across a dataset and can also be used to resize images to a particular size.

