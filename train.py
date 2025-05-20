import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import model as m
from . import data as d
import os

# 参数
BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.95
LAMBDA_U = 1.0
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# evaluation
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")
    model.train()

# train
def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, device, epochs=50, confidence_threshold=0.95, lambda_u=1.0, ckpt_path="./checkpoints"):
    os.makedirs(ckpt_path, exist_ok=True)
    model.train()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for (xl, yl), (xu_w, xu_s) in zip(labeled_loader, unlabeled_loader):
            xl, yl = xl.to(device), yl.to(device)
            xu_w, xu_s = xu_w.to(device), xu_s.to(device)

            logits_l = model(xl)
            loss_l = ce_loss(logits_l, yl)

            with torch.no_grad():
                pseudo_labels = F.softmax(model(xu_w), dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(confidence_threshold).float()

            logits_u = model(xu_s)
            loss_u = (F.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean()

            loss = loss_l + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
        evaluate(model, d.test_loader, device)

        # Save checkpoint
        ckpt_file = os.path.join(ckpt_path, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_file)

# 启动程序
def run(model_name='cnn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader = DataLoader(d.labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(d.unlabeled_dataset, batch_size=64, shuffle=True)

    model_dict = {
        'cnn': m.SimpleCNN,
        'mlp': m.MLP,
        'deepcnn': m.DeepCNN,
        'resnet': m.ResNetLike,
        'lenet': m.TinyLeNet
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model_dict[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, device, EPOCHS, CONFIDENCE_THRESHOLD, LAMBDA_U)
