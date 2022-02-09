import numpy as np
import cv2
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from earlystopping import EarlyStopping
import matplotlib.pyplot as plt
import model
from net_parameters import *
import utils
import torch.nn.functional as F


def train(train_data, valid_data, args, device):
    # model 결과 log dir
    log_dir = "./saver"
    log_heads = ['epoch', 'val_loss']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "log.csv"), 'w') as f:
        f.write(','.join(log_heads)+ '\n')

    train_loader, val_loader = dataset.data_loader(train_data, valid_data, args.batch_size)

    # Seq2Seq -> nn.Sequential 클래스를 이용해 Multi Layer 구성을 만듬.
    endtoendmodel = model.Ensemble(convLSTM_parameters_list , num_layers=3, transpose_channels_list=transpose_channels_list).to(device)

    early_stopping = EarlyStopping(patience=5, improved_valid=True)


    if os.path.exists(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
        endtoendmodel.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        cur_epoch = 0


    optimizer = Adam(endtoendmodel.parameters(), lr=args.lr)

    loss_function = nn.MSELoss()


    for epoch in range(cur_epoch, args.max_epoch+1):
        train_loss = 0
        for batch_num, (input, target) in tqdm(enumerate(train_loader, 1)):
            optimizer.zero_grad()
            endtoendmodel.train()
            input, target = input.to(device), target.to(device)
            output = endtoendmodel(input)
            loss = torch.sqrt(loss_function(output, target))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        val_loss = 0
        endtoendmodel.eval()
        with torch.no_grad():
            for input, target in tqdm(val_loader):
                input, target = input.to(device), target.to(device)
                output = endtoendmodel(input)
                loss = torch.sqrt(loss_function(output, target))
                val_loss += loss

        val_loss /= len(val_loader.dataset)

        print(f"Epoch: {epoch} Training Loss: {train_loss}, Validation Loss: {val_loss}")

        model_dict = {
            'epoch': epoch,
            'state_dict': endtoendmodel.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        early_stopping(val_loss.item(), model_dict, epoch, args.save_dir, log_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def inference(test_data, args, device):

    test_loader = dataset.data_loader_test(test_data, args.batch_size)

    endtoendmodel = model.Ensemble(convLSTM_parameters_list , num_layers=3, transpose_channels_list=transpose_channels_list).to(device)

    if os.path.exists(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
        endtoendmodel.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(endtoendmodel.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1

    batch, target = next(iter(test_loader))

    output = np.zeros(target.shape, dtype=np.uint8)

    target = target * 255.

    # Loop over timesteps 5~10 sequence data prediction.
    for timestep in range(target.shape[2]):
        input = batch[:, :, timestep:timestep + 5]
        output[:, :, timestep] = endtoendmodel(input).cpu().detach().numpy() * 255.0

        metric = utils.mse(output[:, :, timestep], target[:, :, timestep])

        print(metric)

    save_path = os.path.dirname(os.path.abspath(__file__))

    utils.visualization(target, output, save_path)


# 나중에 segmentation 할때
# target = F.one_hot(target.long(), 2)
# segmentation일때
# acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(target[:, :, timestep], output[:, :, timestep], 2)
# print(f"sequence:{timestep+1} - acc:{acc}, acc_cls:{acc_cls}, mean_iou:{mean_iu}")
