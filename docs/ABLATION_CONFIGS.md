# LunaVLA 扩展实验配置

## 学习率消融实验

### configs/act_pusht_lr_ablation_1e5.yaml
learning_rate: 1.0e-5
# 其他参数与 baseline 相同

### configs/act_pusht_lr_ablation_5e5.yaml  
learning_rate: 5.0e-5

### configs/act_pusht_lr_ablation_1e4.yaml
learning_rate: 1.0e-4

### configs/act_pusht_lr_ablation_5e4.yaml
learning_rate: 5.0e-4

### configs/act_pusht_lr_ablation_1e3.yaml
learning_rate: 1.0e-3


## Batch Size 消融实验

### configs/act_pusht_bs_ablation_8.yaml
batch_size: 8

### configs/act_pusht_bs_ablation_16.yaml
batch_size: 16

### configs/act_pusht_bs_ablation_32.yaml
batch_size: 32

### configs/act_pusht_bs_ablation_64.yaml
batch_size: 64


## 编码器深度消融实验

### configs/act_pusht_depth_ablation_2.yaml
encoder_depth: 2

### configs/act_pusht_depth_ablation_4.yaml
encoder_depth: 4

### configs/act_pusht_depth_ablation_6.yaml
encoder_depth: 6

### configs/act_pusht_depth_ablation_8.yaml
encoder_depth: 8
