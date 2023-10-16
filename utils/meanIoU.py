import numpy as np
import torch

def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape,
              ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone(
        ).detach().cpu().numpy().argmax(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else:
            iou_score = intersection / union
        iousum += iou_score

    miou = iousum/target.shape[0]
    return miou

def marco_iou(outputs, targets, batch_size = 16, n_classes = 13):
    """Intersection over union
        Ref from: https://github.com/hayashimasa/UNet-PyTorch/blob/main/metric.py
    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
        n_classes (int): number of segmentation classes
    """
    eps = 1e-6
    class_iou = np.zeros(n_classes)

    outputs = torch.argmax(outputs, dim=1)
    outputs = outputs.view(batch_size, -1)

    targets = torch.argmax(targets, dim=1)
    targets = targets.view(batch_size, -1)

    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()
        targets_cpu = targets[idx].cpu()

        for c in range(n_classes):
            i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
            i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
            intersection = np.intersect1d(i_outputs, i_targets).size
            union = np.union1d(i_outputs, i_targets).size
            class_iou[c] += (intersection + eps) / (union + eps)

    class_iou /= batch_size

    return class_iou.mean()
