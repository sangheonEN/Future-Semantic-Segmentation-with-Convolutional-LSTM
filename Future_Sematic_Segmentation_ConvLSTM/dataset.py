import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from requests import get
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_set():
    # Load Data as Numpy Array
    # mnist_sequence_data url: http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
    if not os.path.exists(os.path.join(BASE_DIR, "mnist_test_seq.npy")):
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        file_name = url.split("/")[-1]

        with open(file_name, "wb") as file:
            response = get(url)
            file.write(response.content)

    MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)

    # Shuffle Data
    np.random.shuffle(MovingMNIST)

    # Train, Test, Validation splits
    train_data = MovingMNIST[:8000]
    val_data = MovingMNIST[8000:9000]
    test_data = MovingMNIST[9000:10000]

    return train_data, val_data, test_data


def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:,:,rand-10:rand], batch[:,:,rand]

def collate_test(batch):

    # Last 10 frames are target
    target = np.array(batch)[:, 10:]

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    return batch, target


# Training Data Loader
# collate의 출력 기준으로 각 data loader에 input과 target을 저장.
def data_loader(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data, shuffle=True,
                            batch_size=batch_size, collate_fn=collate)

    # Validation Data Loader
    val_loader = DataLoader(val_data, shuffle=True,
                            batch_size=batch_size, collate_fn=collate)

    return train_loader, val_loader


def data_loader_test(test_data):

    test_loader = DataLoader(test_data, shuffle=True, batch_size=3, collate_fn=collate_test)

    return test_loader

"""
# Debug Get a batch
if __name__ == "__main__":

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data as Numpy Array
    # mnist_sequence_data url: http://www.cs.toronto.edu/~nitish/unsupervised_video/
    MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)

    # Shuffle Data
    np.random.shuffle(MovingMNIST)

    # Train, Test, Validation splits
    train_data = MovingMNIST[:8000]
    val_data = MovingMNIST[8000:9000]
    test_data = MovingMNIST[9000:10000]

    train_loader, val_loader = data_loader(train_data, val_data)

    # input -> 0~9번째 시퀀스 학습 input 이미지 데이터 취합한것 그래서 shape (batch, channel, seq_num, height, width)
    # _ -> 10번째 정답 target 이미지 데이터 shape (batch, channel, height, width)
    input, _ = next(iter(val_loader))

    # Reverse process before displaying
    input = input.cpu().numpy() * 255.0

    for video in input.squeeze(1)[:3]:
        p = np.transpose(video.astype(np.int8), (1, 2, 0))
        for i in range(p.shape[2]):
            cv2.imwrite(f"./data/{i}_image.png", p[:, :, i])
"""
