# toySSL, Semi-Supervised Learning Framework on MNIST & EMNIST

本项目实现了一个基于 PyTorch 的半监督学习框架，利用 FixMatch 思想对 MNIST 和 EMNIST 数据集进行训练和测试。支持多种轻量及主流神经网络模型，包括 CNN、ResNet、MobileNet 等。

## 目录

- [项目简介](#项目简介)
- [代码结构](#代码结构)
- [快速开始](#快速开始)
- [半监督数据集构造](#半监督数据集构造)
- [模型列表](#模型列表)
- [训练细节](#训练细节)
- [实验结果](#实验结果)

---

## 项目简介

本项目通过结合 MNIST 和 EMNIST 数据集，实现了半监督学习中的 FixMatch 算法核心思想：

- **少量有标签数据**（每个类别20张）
- **大量无标签数据**（包含 MNIST 剩余以及 EMNIST 字母子集）
- **伪标签和一致性正则化**
- **动态自适应阈值调整和熵最小化损失**

支持多种模型训练与评估，方便实验和学习半监督学习技术。

---

## 代码结构

```
.
├── data.py           # 数据处理，数据增强，半监督数据集构建
├── model.py          # 多种模型定义（SimpleCNN, MLP, ResNet18 等）
├── train.py          # 训练主流程，实现 FixMatch 算法及训练逻辑
├── README.md         # 项目说明文件
└── data/             # 存储 MNIST 和 EMNIST 数据
```

---

## 快速开始

### 下载代码

```bash
git clone https://github.com/baihuaDaDa/toySSL.git
cd toySSL
```

### 训练模型

可以直接运行 `train.py` 中的 `run` 函数启动训练：

```bash
python train.py
```

默认训练 ResNet18 模型，使用半监督 FixMatch 训练。

### 指定模型训练

编辑 `train.py` 中的 `run` 函数，调用：

```python
run('cnn')           # 训练简单CNN模型
run('mlp')           # 训练MLP模型
run('resnet')        # ResNet18
run('mobilenet')     # MobileNetV2
run('efficientnet')  # EfficientNetB0
run('deepcnn')       # 深度CNN模型
run('lenet')         # 简单TinyLeNet
run('regnet')        # RegNetY8GF
run('shufflenet')    # ShuffleNetV2_x0_5
```

或者在代码里改成：

```python
if __name__ == '__main__':
    run('mobilenet')
```

运行即可。

---

## 半监督数据集构造

- 有标注数据：MNIST 每类别 20 张图像
- 无标注数据：MNIST 剩余图像 + EMNIST 字母子集（2000张）
- 同时使用弱增强（随机裁剪），强增强（RandAugment）

---

## 模型列表

| 模型名称    | 说明                       |
|-------------|----------------------------|
| SimpleCNN   | 简单三层卷积神经网络       |
| MLP         | 多层感知机                 |
| DeepCNN     | 更深层卷积网络             |
| TinyLeNet   | LeNet 变体                 |
| ResNet18    | ResNet-18 经典残差网络     |
| MobileNetV2 | 轻量级 MobileNetV2         |
| EfficientNetB0 | EfficientNet-B0           |
| RegNetY8GF  | RegNet                     |
| ShuffleNetV2| ShuffleNetV2_x0_5          |

---

## 训练细节

- 优化器：SGD + Nesterov Momentum
- 学习率策略：基于余弦衰减
- 训练周期：默认 100 轮
- 半监督损失权重 `lambda_u=1.0`
- 置信度阈值：动态自适应更新
- 伪标签可信度阈值：默认0.95
- 采用 EMA 模型参数更新以提升稳定性（从第80 epoch开始）
- 支持 Curriculum Batch Size 调整

训练过程中会自动保存loss/acc曲线图和最终准确率日志。

---

## 实验结果

在 MNIST + EMNIST 半监督实验中，ResNet18 能达到较高准确率（具体表现请参考训练日志和最终生成的图片）。
