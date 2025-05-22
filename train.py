import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import model as m
import data as d
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime

# 参数
BATCH_SIZE = 64
MU = 7
CONFIDENCE_THRESHOLD = 0.95
LAMBDA_U = 1.0
EPOCHS = 500
LR = 0.03
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

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
    return acc

# train
def train_fixmatch(model, model_name, labeled_loader, unlabeled_loader, optimizer, device, epochs=50, tau=0.95, lambda_u=1.0, num_classes=10, ckpt_root="./checkpoints", log_root="./logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ckpt_path = os.path.join(ckpt_root, model_name, timestamp)
    # os.makedirs(ckpt_path, exist_ok=True)
    log_path = os.path.join(log_root, model_name)
    os.makedirs(log_path, exist_ok=True)
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    ema_model = deepcopy(model)
    alpha = 0.999
    total_steps = epochs * min(len(labeled_loader), len(unlabeled_loader))
    global_step = 0
    initial_lr = optimizer.param_groups[0]['lr']

    step_log, eval_log, losses, labeled_losses, unlabeled_losses = [], [], [], [], []
    final_acc = 0

    for epoch in range(epochs):
        total_loss = 0
        for (xl, yl), (xu_w, xu_s) in zip(labeled_loader, unlabeled_loader):
            xl, yl = xl.to(device), yl.to(device)
            xu_w, xu_s = xu_w.to(device), xu_s.to(device)

            # cosine decay
            lr = initial_lr * np.cos(7 * np.pi * global_step / (16 * total_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            global_step += 1

            # loss calculation
            logits_l = model(xl)
            loss_l = ce_loss(logits_l, yl)

            with torch.no_grad():
                pseudo_labels = F.softmax(model(xu_w), dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(tau).float()

            logits_u = model(xu_s)
            loss_u = (F.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean()

            loss = loss_l + lambda_u * loss_u

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data = alpha * ema_p.data + (1 - alpha) * model_p.data

            step_log.append((global_step, loss.item(), loss_l.item(), loss_u.item(), lr))
            total_loss += loss.item()
            losses.append(loss.item())
            labeled_losses.append(loss_l.item())
            unlabeled_losses.append(loss_u.item())

        acc = evaluate(ema_model, d.test_loader, device)
        eval_log.append((epoch + 1, acc))
        final_acc = acc

        # Save checkpoint
        # ckpt_file = os.path.join(ckpt_path, f"model_epoch{epoch+1}.pt")
        # torch.save(ema_model.state_dict(), ckpt_file)
    
    # Save final accuracy to a file
    with open(os.path.join(log_root, "acc.txt"), "w") as f:
        f.write(f"{model_name}[{timestamp}]: {final_acc:.2f}\n")

    # 可视化
    steps, loss_list, loss_l_list, loss_u_list, lrs = zip(*step_log)
    epochs_, accs = zip(*eval_log)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(steps, loss_list, label="Total Loss")
    plt.plot(steps, loss_l_list, label="Labeled Loss")
    plt.plot(steps, loss_u_list, label="Unlabeled Loss")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(steps, lrs)
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_, accs)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(log_path, f"{timestamp}.png"))

# 启动训练
def run(model_name='cnn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader = DataLoader(d.labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(d.unlabeled_dataset, batch_size=int(MU * BATCH_SIZE), shuffle=True)

    model_dict = {
        'cnn': m.SimpleCNN,
        'mlp': m.MLP,
        'deepcnn': m.DeepCNN,
        'lenet': m.TinyLeNet,
        'resnet': m.ResNet18,
        'mobilenet': m.MobileNetV2,
        'efficientnet': m.EfficientNetB0,
        'regnet': m.RegNetY8GF,
        'shufflenet': m.ShuffleNetV2,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model_dict[model_name]().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

    train_fixmatch(model, model_name, labeled_loader, unlabeled_loader, optimizer, device, EPOCHS, CONFIDENCE_THRESHOLD, LAMBDA_U)

run('cnn')