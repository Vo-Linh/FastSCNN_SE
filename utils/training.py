import torch
import numpy as np

def train_one_epoch(epoch_index, training_loader, validation_loader, model, optimizer, 
                    scheduler, loss_fn, eval_fn, device):
    running_loss = 0.
    last_loss = 0.
    print(f"Epoch {epoch_index + 1}")
    print(f"Lr: {optimizer.param_groups[0]['lr']}")
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.type(torch.cuda.FloatTensor)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (i + 1)  # loss per batch
    print(f'Training loss: {last_loss:.4f}')
    scheduler.step()
    running_loss = 0.

    with torch.no_grad():
        loss_val = 0
        iou = 0
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            # Make predictions for this batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            outputs = model(inputs)
            loss_val += loss_fn(outputs, labels).item()
            
            iou += eval_fn(outputs, labels.type(torch.cuda.LongTensor))

        mean_iou = iou / len(validation_loader)
        loss_val = loss_val / len(validation_loader)
        print(f"Loss: {loss_val:.4f} IoU: {mean_iou:.4f}")

    return mean_iou

