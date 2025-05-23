
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST, EMNIST
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import random

# 参数
NUM_LABELED = 20

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

# 数据增强
weak_transform = T.Compose([
    # T.RandomHorizontalFlip(),
    T.RandomCrop(28, padding=4),
    T.ToTensor(),
])

strong_transform = T.Compose([
    # T.RandomHorizontalFlip(),
    T.RandomCrop(28, padding=4),
    T.RandAugment(num_ops=2, magnitude=5),
    T.ToTensor(),
])

# 自定义 Dataset
class UnlabeledDataset(Dataset):
    def __init__(self, base_dataset, transform_weak=None, transform_strong=None):
        self.dataset = base_dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return self.transform_weak(img), self.transform_strong(img)

# 构造数据集
def get_train_dataset():
    def get_label_indices(dataset, num_per_class):
        label_indices = []
        for class_id in range(10):
            indices = np.where(np.array(dataset.targets) == class_id)[0]
            chosen = np.random.choice(indices, num_per_class, replace=False)
            label_indices.extend(chosen)
        return label_indices
    
    mnist_weak = MNIST(root="./data", train=True, download=True, transform=weak_transform)

    labeled_idx = get_label_indices(mnist_weak, NUM_LABELED)
    labeled_dataset = Subset(mnist_weak, labeled_idx)

    mnist_full = MNIST(root="./data", train=True, download=True, transform=None)
    emnist_letters = EMNIST(root="./data", split="letters", train=True, download=True, transform=None)

    unlabeled_mnist = Subset(mnist_full, list(set(range(len(mnist_full))) - set(labeled_idx)))
    unlabeled_mnist = Subset(unlabeled_mnist, np.random.choice(len(unlabeled_mnist), 8000, replace=False))

    emnist_subset = Subset(emnist_letters, np.random.choice(len(emnist_letters), 2000, replace=False))

    combined_unlabeled_data = ConcatDataset([unlabeled_mnist, emnist_subset])
    unlabeled_dataset = UnlabeledDataset(combined_unlabeled_data, transform_weak=weak_transform, transform_strong=strong_transform)

    return labeled_dataset, unlabeled_dataset

def get_test_dataset():
    return MNIST(root="./data", train=False, transform=T.ToTensor(), download=True)