import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def mse(pred, target):
    mse_value = mean_absolute_error(target.flatten(), pred.flatten())

    return mse_value


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask],
                       minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(target, pred, num_class):
    """Returns accuracy score evaluation result.
        https://gaussian37.github.io/vision-segmentation-miou/
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc

    """
    target = np.average(target, axis=1) / 255.
    pred = np.average(pred, axis=1) / 255.

    hist = np.zeros((num_class, num_class))
    for lt, lp in zip(target.astype(int), pred.astype(int)):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def image_plot(target_arr, output_arr):
    fig = plt.figure()
    rows = 2
    cols = 5

    for i in range(rows):
        for j in range(cols):
            if i == 0:
                ax1 = fig.add_subplot(rows, cols, i + j+1)
                ax1.imshow(target_arr[j])
                ax1.set_title(f"{j}_target")
                ax1.axis("off")
            else:
                ax1 = fig.add_subplot(rows, cols, i+cols-1 + j+1)
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
        tgt = np.average(tgt, axis=0)
        out = np.average(out, axis=0)
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