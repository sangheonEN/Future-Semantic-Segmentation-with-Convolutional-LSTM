import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, improved_valid=False):
        """
        이전의 validation loss를 비교하여 개선되는 epoch가 없으면 일찍 학습 중단.

        :param patience: 참는 수. ex patience=6 한 6번까지 개선안되도 스탑하진 않는다. (int)
        :param improved_valid: 개선된 validation loss 출력 할지말지 (bool)
        """
        self.patience = patience
        self.improved_valid = improved_valid
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        # score는 validation loss의 음수 값을 사용 부호 반대 고려하기.
        score = -val_loss

        if self.best_score is None:
            # epoch 1일때, 초기 best score 저장
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            # new score가 이전 self.best_score보다. 실제 val_loss 부호 반대니까 작으면 크다고 생각해야함.
            # 따라서, new score가 이전 best_score보다 크니까 더 안좋은 학습 결과임. 카운트함.
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0


    def save_checkpoint(self, val_loss, model, epoch, save_path):
        """
        Saves model when validation loss decrease.

        :param val_loss: improved val_loss
        :param model: network model
        :param epoch: current epoch
        :param save_path: -
        """
        if self.improved_valid:
            print(f"validation loss decreased {self.val_loss_min:.6f} --> {val_loss:.6f}. Saving Model")

        torch.save(model, os.path.join(save_path, f"checkpoint.pth.tar"))

        self.val_loss_min = val_loss
