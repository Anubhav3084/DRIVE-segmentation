
import os
import torch
import torch.onnx as onnx
import logging
import argparse
from tqdm import tqdm
from model import UNet
from torch.optim import Adam, SGD
from utils import *

import wandb as wb


def trainEpoch(model, train_dl, optimizer, loss_fn, device):
    
    epoch_loss = 0.0
    model.train()
    model.to(device)
    for i, (img, label) in enumerate(train_dl):
        x, y = img.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(train_dl)
    return epoch_loss

def evaluateEpoch(model, valid_dl, loss_fn, device):
    epoch_loss = 0.0
    dice_score = 0.0

    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (img, label) in enumerate(valid_dl):
            x, y = img.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            dice_score += DiceScore(y_pred, y)

        epoch_loss = epoch_loss/len(valid_dl)
    return epoch_loss, dice_score / len(valid_dl)

def train(train_dl, valid_dl, args, logger):

    if args.model == 'unet':
        model = UNet(X=64)
    else:
        raise NotImplementedError
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    
    if args.loss_fn == 'bce':
        loss_fn = BCELoss
    elif args.loss_fn == 'dice_bce':
        loss_fn = DiceBCELoss
    else:
        raise NotImplementedError

    wb.watch(model, loss_fn, log='all', log_freq=10)

    training_loss, validation_loss, dice_scores = [], [], []
    best_epoch_valid_loss = float('inf')

    for epoch in tqdm(range(args.epochs)):

        epoch_train_loss = trainEpoch(model, train_dl, optimizer, loss_fn, args.device)
        epoch_valid_loss, dice_score = evaluateEpoch(model, valid_dl, loss_fn, args.device)
        
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_valid_loss)
        dice_scores.append(dice_score)

        wb.log({
            'epoch': epoch,
            'train loss': epoch_train_loss,
            'valid loss': epoch_valid_loss,
            'dice score': dice_score
        })
        
        """ Saving the model """
        if epoch_valid_loss < best_epoch_valid_loss:
            best_epoch_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), args.savePath+'best_model_state_dict.pt')
        
        logger.info(f'Epoch: {epoch+1}/{args.epochs}\t train loss: {round(epoch_train_loss, 4)}, valid loss: {round(epoch_valid_loss, 4)}, dice score: {round(dice_score, 4)}')

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), args.savePath+'last_model_state_dict.pt')
            torch.save(training_loss, args.savePath+'training_loss.pt')
            torch.save(validation_loss, args.savePath+'validation_loss.pt')
            torch.save(dice_scores, args.savePath+'dice_scores.pt')

    torch.save(model.state_dict(), args.savePath+'last_model_state_dict.pt')
    torch.save(training_loss, args.savePath+'training_loss.pt')
    torch.save(validation_loss, args.savePath+'validation_loss.pt')
    torch.save(dice_scores, args.savePath+'dice_scores.pt')

    logger.info('-------->>> Training completed <<<--------')

def inference(test_dl, args, logger):

    if args.model == 'unet':
        model = UNet(X=64)
        model.load_state_dict(torch.load(args.savePath+'best_model_state_dict.pt'))
    else:
        raise NotImplementedError
    
    if args.loss_fn == 'bce':
        loss_fn = BCELoss
    elif args.loss_fn == 'dice_bce':
        loss_fn = DiceBCELoss
    else:
        raise NotImplementedError

    test_loss, test_dice = evaluateEpoch(model, test_dl, loss_fn, args.device)
    wb.log({
        'test dice score': test_dice
    })
    logger.info(f'Test results -> loss: {round(test_loss, 4)}, dice score: {round(test_dice, 4)}')

    test_results = {
        'test_loss': test_loss,
        'test_dice': test_dice
    }
    torch.save(test_results, args.savePath+'test_results.pt')

if __name__ == '__main__':
    pass