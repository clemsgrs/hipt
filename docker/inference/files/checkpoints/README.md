# Model Checkpoints

This folder should contain two subfolders that store  model checkpoints for running inference:

* `pretrained`: dump in this folder the weights of the SSL-pretrained `ViT_patch` (for each fold)
* `trained`: dump in this folder the weights for the trained Local H-ViT (for each fold)

In the end, the `checkpoints` folder structure should be:

```
checkpoints/
├── pretrained/
│     ├── vit_256_small_dino_fold_0.pt
│     ├── vit_256_small_dino_fold_1.pt
│     ├── vit_256_small_dino_fold_2.pt
│     ├── vit_256_small_dino_fold_3.pt
│     └── vit_256_small_dino_fold_4.pt
├── trained/
│     ├── fold_0.pt
│     ├── fold_1.pt
│     ├── fold_2.pt
│     ├── fold_3.pt
│     └── fold_4.pt
```
