# pointnet-autoencoder-pytorch
Implementaion of Auto Encoder in Pytorch forked from 
```
https://github.com/dhirajsuvarna/pointnet-autoencoder-pytorch
```

# Dataset
Data currently tested on modelnet40, dataloader is ultilized from
```
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
```

# Command line to run

```
python train.py --model_type fxia --num_points 1024 --dataset_path data/modelnet40/modelnet40_normal_resampled --nepoch 200
python train_acronym.py --model_type fxia --num_points 1024 --dataset_path data/acronym --nepoch 200
```

# Command line to infer

```
python infer_modelnet.py --input_folder data/modelnet40/modelnet40_normal_resampled --nn_model saved_models/autoencoder_fxia_193.pth --nn_model_type fxia --num_points 1024 
python infer_acronym.py --input_folder data/acronym --nn_model saved_models/autoencoder_acronym_fxia_2.pth --nn_model_type fxia --num_points 1024 
```

# Command for Tensor board check

```
tensorboard --logdir=runs
```
