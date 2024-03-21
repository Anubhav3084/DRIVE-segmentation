

import os
import torch
import numpy as np
import torch.nn.functional as F

def DiceBCELoss(preds, targets, smooth=1):
    
    preds = torch.softmax(preds, dim=1)   # since model don't have a sigmoid activation
    
    # flatten label and predictions
    preds = preds.view(-1)     
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice_loss = 1 - ((2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth))
    BCE = F.binary_cross_entropy(preds, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

def BCELoss(preds, targets):

    preds = torch.softmax(preds, dim=1)
    preds = preds.view(-1)     
    targets = targets.view(-1)
    BCE = F.binary_cross_entropy(preds, targets, reduction='mean')
    return BCE

def DiceScore(preds, targets, smooth=1):
    
    preds = torch.softmax(preds, dim=1)   # since model don't have a sigmoid activation
    
    # flatten label and predictions
    preds = preds.view(-1)     
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice_score = ((2. * intersection + smooth) / (preds.sum() + targets.sum()) + smooth)
    
    return dice_score.item()

def compute_metrics():
    pass