import numpy as np
import cv2
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import convlstm as cl
from earlystopping import EarlyStopping
import matplotlib.pyplot as plt
import model
from net_parameters import *


def image_plot(target_arr, output_arr):
    fig = plt.figure()
    rows = 2
    cols = 10

    for i in range(rows):
        for j in range(cols):
            if i == 0:
                ax1 = fig.add_subplot(rows, cols, i + j+1)
                ax1.imshow(target_arr[j])
                ax1.set_title(f"{j}_target")
                ax1.axis("off")
            else:
                ax1 = fig.add_subplot(rows, cols, i+9 + j+1)
                ax1.imshow(output_arr[j])
                ax1.set_title(f"{j}output")
                ax1.axis("off")

    plt.show()
    print("출력되나?")



def visualization(target, output, save_path):
    import io
    import imageio
    from ipywidgets import widgets, HBox

    pred_array = []
    target_array = []

    for tgt, out in zip(target, output):  # Loop over samples
        # target, output shape: (batch, sequence, height, width)
        # tgt, out shape: (sequence, height, width)
        for i in range(tgt.shape[0]):
            img_tgt = np.expand_dims(tgt[i,:], -1)
            img_out = np.expand_dims(out[i,:], -1)
            cv2.imwrite(os.path.join(save_path, f"target/{i}_image_target.png"), img_tgt)
            cv2.imwrite(os.path.join(save_path, f"pred/{i}_image_prediction.png"), img_out)
            pred_array.append(img_out)
            target_array.append(img_tgt)
        image_plot(target_array, pred_array)


def train(train_data, valid_data, args, device):

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

    loss_function = nn.BCEWithLogitsLoss(reduction='sum')


    for epoch in range(cur_epoch, args.max_epoch+1):
        train_loss = 0
        endtoendmodel.train()
        for batch_num, (input, target) in enumerate(train_loader, 1):
            input, target = input.to(device), target.to(device)
            output = endtoendmodel(input)
            loss = loss_function(output.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        val_loss = 0
        endtoendmodel.eval()
        with torch.no_grad():
            for input, target in val_loader:
                output = endtoendmodel(input)
                loss = loss_function(output.flatten(), target.flatten())
                val_loss += loss

        val_loss /= len(val_loader.dataset)

        print(f"Epoch: {epoch} Training Loss: {train_loss}, Validation Loss: {val_loss}")

        model_dict = {
            'epoch': epoch,
            'state_dict': endtoendmodel.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        early_stopping(val_loss.item(), model_dict, epoch, args.save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def inference(test_data, args, device):

    test_loader = dataset.data_loader_test(test_data)

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

    # Loop over timesteps
    for timestep in range(target.shape[1]):
        input = batch[:, :, timestep:timestep + 10]
        output[:, timestep] = (endtoendmodel(input).squeeze(1).cpu() > 0.5) * 255.0

    save_path = os.path.dirname(os.path.abspath(__file__))

    visualization(target, output, save_path)


