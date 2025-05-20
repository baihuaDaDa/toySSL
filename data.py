
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST, EMNIST
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from torchvision.transforms.autoaugment import RandAugment

# 参数
NUM_LABELED = 250

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_all()

# 数据增强
weak_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(28, padding=4),
    T.ToTensor()
])

strong_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(28, padding=4),
    RandAugment(num_ops=2, magnitude=5),
    T.ToTensor()
])

# 自定义 Dataset
class SSLDataset(Dataset):
    def __init__(self, base_dataset, transform_weak=None, transform_strong=None):
        self.data = base_dataset.data
        self.targets = base_dataset.targets
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].numpy()
        img = np.expand_dims(img, axis=2)
        img = T.ToPILImage()(img)

        if self.transform_weak and self.transform_strong:
            weak_img = self.transform_weak(img)
            strong_img = self.transform_strong(img)
            return weak_img, strong_img
        else:
            return self.transform_weak(img), self.targets[idx]

def get_label_indices(dataset, num_per_class):
    label_indices = []
    for class_id in range(10):
        indices = np.where(np.array(dataset.targets) == class_id)[0]
        chosen = np.random.choice(indices, num_per_class, replace=False)
        label_indices.extend(chosen)
    return label_indices

mnist_full = MNIST(root="./data", train=True, download=True)
labeled_idx = get_label_indices(mnist_full, 20)
labeled_dataset = Subset(mnist_full, labeled_idx)
labeled_dataset = SSLDataset(labeled_dataset, transform_weak=weak_transform)

unlabeled_mnist = Subset(mnist_full, list(set(range(len(mnist_full))) - set(labeled_idx)))
unlabeled_mnist = torch.utils.data.Subset(unlabeled_mnist, np.random.choice(len(unlabeled_mnist), 8000, replace=False))

emnist_letters = EMNIST(root="./data", split="letters", train=True, download=True)
emnist_subset = torch.utils.data.Subset(emnist_letters, np.random.choice(len(emnist_letters), 2000, replace=False))

combined_unlabeled_data = torch.utils.data.ConcatDataset([unlabeled_mnist, emnist_subset])
unlabeled_dataset = SSLDataset(combined_unlabeled_data, transform_weak=weak_transform, transform_strong=strong_transform)

test_dataset = MNIST(root="./data", train=False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)