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
EPOCHS = 100
LR = 0.03
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# evaluation
def evaluate(model, test_loader, epoch, device):
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
    print(f"[{epoch}] Test Accuracy: {acc:.2f}%")
    model.train()
    return acc

# threshold update
def threshold_cpl_M(x):
    return x / (2 - x)

def threshold_cpl(max_probs, targets_u, tau, batch_size, min_thres=0.5, fn_M=threshold_cpl_M, num_classes=10, device=device):
    sigma = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        mask_c = targets_u.eq(c) & max_probs.ge(tau)
        sigma[c] = mask_c.sum()
    beta = sigma / max(torch.max(sigma), batch_size - sigma.sum())
    return (fn_M(beta) * tau).clamp(min=min_thres)

def threshold_linear(epoch, k=50, max_t=0.95, min_t=0.5):
    step = (max_t - min_t) / k
    return min(max_t, min_t + (epoch - 1) * step)

def threshold_smooth(epoch, k=50, max_t=0.95, min_t=0.5):
    thres = min_t + (1 - min_t) * (1 - np.exp(-(epoch - 1) / k))
    return min(max_t, thres)

# entropy mean loss
def eml(max_probs, targets_u, logits_u_s, threshold):
    with torch.no_grad():
        # Create mask for samples with confidence below threshold
        softmax_pred = F.softmax(logits_u_s, dim=1)
        # Identify target and non-target samples
        _, num_classes = softmax_pred.shape # batch_size, num_classes
        
        # Create mask for target class (1 for target class, 0 for others)
        target_mask = torch.zeros_like(softmax_pred)
        target_mask.scatter_(1, targets_u.unsqueeze(1), 1)
        
        # Create mask for non-target classes
        non_target_mask = 1 - target_mask
        
        # Mask for samples below threshold but still valuable
        valid_mask = max_probs.lt(threshold).float().unsqueeze(1) * non_target_mask
        
        # Calculate soft multi-label for entropy minimization
        # Target probability is complementary to sum of non-target probabilities
        target_prob = softmax_pred.gather(1, targets_u.unsqueeze(1))
        non_target_prob = (1 - target_prob) / (num_classes - 1)
        soft_target = target_mask * target_prob + non_target_mask * non_target_prob
        
        # Apply entropy minimization only to valid samples
        entropy = -(soft_target * torch.log(softmax_pred + 1e-10) + 
                   (1 - soft_target) * torch.log(1 - softmax_pred + 1e-10))
        
    # Calculate final loss
    loss_eml = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-10)
    return loss_eml

# train
def train_fixmatch(model, model_name, labeled_loader, unlabeled_loader, test_loader, optimizer, device, epochs=50, tau=0.95, lambda_u=1.0, num_classes=10, ckpt_root="./checkpoints", log_root="./logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ckpt_path = os.path.join(ckpt_root, model_name, timestamp)
    # os.makedirs(ckpt_path, exist_ok=True)
    log_path = os.path.join(log_root, model_name)
    os.makedirs(log_path, exist_ok=True)
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    alpha = 0.999
    ema_start = 100
    global_step = 0
    initial_lr = optimizer.param_groups[0]['lr']
    threshold = torch.full((num_classes,), 0.5, device=device)

    step_log, eval_log, losses, labeled_losses, unlabeled_losses = [], [], [], [], []

    for epoch in range(epochs):
        total_loss = 0
        for (xl, yl), (xu_w, xu_s) in zip(labeled_loader, unlabeled_loader):
            xl, yl = xl.to(device), yl.to(device)
            xu_w, xu_s = xu_w.to(device), xu_s.to(device)

            # loss calculation
            logits_l = model(xl)
            loss_l = ce_loss(logits_l, yl)

            logits_u_w = model(xu_w)
            with torch.no_grad():
                pseudo_labels = F.softmax(logits_u_w, dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(threshold[targets_u]).float()

            logits_u_s = model(xu_s)
            loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            # loss_em = eml(max_probs, targets_u, logits_u_s, tau)

            loss = loss_l + lambda_u * loss_u

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            if epoch >= ema_start:
                if epoch == ema_start:
                    ema_model = deepcopy(model)
                with torch.no_grad():
                    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data = alpha * ema_p.data + (1 - alpha) * model_p.data

            # update threshold
            with torch.no_grad():
                threshold = threshold_cpl(max_probs, targets_u, tau, xu_w.size(0))

            global_step += 1
            step_log.append((global_step, loss.item(), loss_l.item(), loss_u.item()))
            total_loss += loss.item()
            losses.append(loss.item())
            labeled_losses.append(loss_l.item())
            unlabeled_losses.append(loss_u.item())

        # cosine decay
        lr = initial_lr * np.cos(7 * np.pi * epoch / (16 * epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        acc = evaluate((model if epoch < ema_start else ema_model), test_loader, epoch, device)
        eval_log.append((epoch + 1, lr, acc))

        # Save checkpoint
        # ckpt_file = os.path.join(ckpt_path, f"model_epoch{epoch+1}.pt")
        # torch.save(ema_model.state_dict(), ckpt_file)
    
    # Save final accuracy to a file
    with open(os.path.join(log_root, "acc.txt"), "a") as f:
        f.write(f"{model_name}[{timestamp}]: {acc:.2f}\n")

    # 可视化
    steps, loss_list, loss_l_list, loss_u_list = zip(*step_log)
    epochs_, lrs, accs = zip(*eval_log)

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
    plt.plot(epochs_, lrs)
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
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

    labeled_dataset, unlabeled_dataset = d.get_train_dataset()
    test_dataset = d.get_test_dataset()

    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=int(MU * BATCH_SIZE), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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

    train_fixmatch(model, model_name, labeled_loader, unlabeled_loader, test_loader, optimizer, device, EPOCHS, CONFIDENCE_THRESHOLD, LAMBDA_U)

run('resnet')