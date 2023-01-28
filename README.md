# Study-on-Loss-Landscape-Geometry-for-Improving-Generalization-in-Adaptive-Optimization-Methods
In this project we analyse the loss landscape geometry of adaptive optimization methods, using a measure of loss landscape curvature called sharpness. It is well known that flat minima are associated with better generalization. Improved methods that aid convergence of adaptive methods, namely ADAM, to flat minima are proposed, for improving their generalization.

![image](https://user-images.githubusercontent.com/40617986/215286069-b049d219-ef70-4e53-b68f-909ec2058d85.png)
Figure 1. Schematic diagram showing intuition behind sharp and flat minima and generalization. Adapted from [1].

<p align="center">
  <img width="600" height="250" src="https://user-images.githubusercontent.com/21705597/175521022-a43d5c96-c474-4105-91ed-370f7a60cd0d.png">
</p>

<p align = "center">
Figure 1. Schematic diagram showing intuition behind sharp and flat minima and generalization. Adapted from [1].
</p>

### Requirements
1. Python 3.6
2. PyTorch
3. Torchvision
4. Matplotlib
5. CUDA (if available)


## Experiments

### a. Baseline
```
python ./baseline/baseline.py
```

### b. SAM
```
python ./sam/sam_train_models.py
```

### c. Plotting
The plots are obtained from the `.npy` files saved from the previous steps using the notebook `plotting.ipynb`.

## Results

The trained models and numpy arrays of metrics (containing train loss, train accuracy, test loss, test accuracy, and sharpness values versus epochs) are stored in this Google Drive [link](https://drive.google.com/drive/folders/1OHBn2H5YkTv_3-dH91hsj9mdSPdAUFhV?usp=sharing). The plots generated from our experiments are also included in this link. In the table given below, we report the mean and standard deviation of test accuracy for all setups.

| Optimizer/Architecture | ResNet18 | VGG16 |
|---|---|---|
| SGD | 90.899 $\pm$ 0.137 | 90.089 $\pm$ 0.070 |
| SGD+SAM | 91.873 $\pm$ 0.102 | 90.453 $\pm$ 0.066 |
| ADAM | 92.716 $\pm$ 0.247 | 92.173 $\pm$ 0.310 |
| ADAM+SAM | 94.896 $\pm$ 0.160 | 94.126 $\pm$ 0.112 |


## Acknowledgement
I would like to acknowledge the following code repositories on which our code is based:

- [loss-landscape](https://github.com/tomgoldstein/loss-landscape)
- [sam](https://github.com/davda54/sam)

## References
[1] Jungmin Kwon, Jeongseop Kim. ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks. November 04, 2021. [blog](https://research.samsung.com/blog/ASAM-Adaptive-Sharpness-Aware-Minimization-for-Scale-Invariant-Learning-of-Deep-Neural-Networks)

