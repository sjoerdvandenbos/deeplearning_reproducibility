import torch.nn as nn
import torch


def batch_get_new_label(old_labels, predictions):
    old_with_padding = nn.ReplicationPad2d(2)(old_labels)
    extended_labels = nn.MaxPool2d(5, stride=1)(old_with_padding)
    extension = extended_labels - old_labels

    # Transforms only the extension area to contain 0 if prediction is 0 and 1 if prediction is 1
    ext_isect_pred = torch.logical_and(extension, predictions)

    # Combines this new extension area with old label
    new_label = torch.logical_or(old_labels, ext_isect_pred).float()
    return new_label


def get_prec_recall(data_loader, network, device):
    network.eval()
    isect_sum = torch.tensor([0], dtype=torch.float32, device=device)
    positive_predicts_pixels = 0
    positive_truth_pixels = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            pred = (network(X) >= 0.5).float()
            y_ext = batch_get_new_label(y, pred)
            isect_sum += torch.sum(torch.logical_and(y_ext, pred))
            positive_predicts_pixels += torch.sum(pred)
            positive_truth_pixels += torch.sum(y_ext)

    precision = (isect_sum + 1e-8) / (positive_predicts_pixels + 1e-8)
    recall = (isect_sum + 1e-8) / (positive_truth_pixels + 1e-8)
    return (precision.item(), recall.item())


def get_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)
