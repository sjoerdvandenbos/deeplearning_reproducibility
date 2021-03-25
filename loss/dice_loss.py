import torch


def batch_dice_loss(true_val, pred_val, epsilon=1e-8):
    """
    Dice coefficient loss
    ---
    Equivalent to F1 score. Often used when there is
    imbalance in dataset (non-crack pixels outnumber
    crack pixels by 65:1).

    Args:
        true_val: a tensor of shape [N, 1, H, W]
        predicted_val: a tensor of shape [N, 1, H, W]
    Returns:
        dice_loss: the Dice loss.
    """
    # Sigmoid -> [0, 1], reflect probabilities
    pred_val = (torch.sigmoid(pred_val) >= 0.5).float()

    # Flattened from [N, 1, H, W] to [N, H*W]
    true_val = true_val.flatten(start_dim=1)
    pred_val = pred_val.flatten(start_dim=1)

    numerator = 2. * (pred_val * true_val).sum(dim=1)
    denominator = (pred_val).sum(dim=1) + (true_val).sum(dim=1)

    return torch.mean(1 - ((numerator + epsilon) / (denominator + epsilon)))


def dice_loss(true_val, pred_val, epsilon=1e-8):
    """
    Dice coefficient loss
    ---
    Equivalent to F1 score. Often used when there is
    imbalance in dataset (non-crack pixels outnumber
    crack pixels by 65:1).

    Args:
        true_val: a tensor of shape [N, 1, H, W]
        predicted_val: a tensor of shape [N, 1, H, W]
    Returns:
        dice_loss: the Dice loss.
    """
    # Sigmoid -> [0, 1], reflect probabilities
    pred_val = (torch.sigmoid(pred_val) >= 0.5).float()

    # Flattened from [H, W] to [H*W]
    true_val = true_val.flatten()
    pred_val = pred_val.flatten()

    numerator = 2. * (pred_val * true_val).sum()
    denominator = (pred_val).sum() + (true_val).sum()

    return torch.mean(1 - ((numerator + epsilon) / (denominator + epsilon)))

#
# prediction = torch.randint(low=-255, high=256, size=(3, 3), dtype=torch.float)

# pred_is_truth = (prediction >= 0).float()
# pred_is_false = (prediction < 0).float()
# pred_is_half = torch.flip(pred_is_truth, [0, 1])

# print(pred_is_truth, "\n\n", pred_is_false, "\n\n", pred_is_half)

# print(f"different tensors should output 1: {dice_loss(pred_is_false, prediction)}")
# print(f"the opposite tensors should output 0: {dice_loss(pred_is_truth, prediction)}")
# print(f"the avg tensors should output [0,1]: {dice_loss(pred_is_half, prediction)}")
