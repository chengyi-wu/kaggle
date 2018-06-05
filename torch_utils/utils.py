import torch
import torchvision
import time
from datetime import timedelta
import numpy as np


def evaluate(model, loader):
    model.eval()
    log_preds = []
    preds = []
    for x, _ in loader:
        scores = model(x)
        log_preds += scores.data.tolist()
        _, y_preds = scores.data.max(1)
        preds += y_preds.tolist()
    return np.array(log_preds), np.array(preds)

def validate(model, val_loder, loss_fn):
    model.eval()
    num_samples = 0
    num_correct = 0
    val_loss = 0
    for x, y in val_loder:
        scores = model(x)
        _, y_preds = scores.data.max(1)
        loss = loss_fn(scores, y)
        val_loss += loss.item()
        num_correct += (y_preds == y).sum()
        num_samples += len(x)
    val_acc = float(num_correct) / num_samples
    val_loss /= num_samples
    return val_loss, val_acc

def train(model, train_loder, loss_fn, optimizer, num_epoches = 1, val_loader = None):
    for epoch in range(num_epoches):
        epoch_start = time.time()
        num_samples = 0
        num_correct = 0
        train_loss = 0
        model.train()
        for x, y in train_loder:

            scores = model(x)
            _, y_preds = scores.data.max(1)

            loss = loss_fn(scores, y)
            train_loss += loss.item()

            num_samples += len(x)
            num_correct += (y_preds == y).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc = float(num_correct) / num_samples
        train_loss /= num_samples
        duration = timedelta(seconds=time.time() - epoch_start)
        stats = "[Epoch %d / %d] train_loss = %.4f, train_acc = %.2f" % (
                epoch + 1, num_epoches,
                train_loss, train_acc * 100,
                )

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, loss_fn)
            print("%s, val_loss = %.4f, val_acc = %.2f, duration = %s"  % (
                stats,
                val_loss, val_acc * 100,
                duration
            ))
        else:
            print("%s, duration = %s" % (stats, duration))

        