import torch
import torch.nn as nn


# pred 320*320*n_class label 320*320*n_class
def BCE_loss(pred,label):
    bce_loss = nn.BCELoss(size_average=True)
    bce_out = bce_loss(pred, label)
    # print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out
