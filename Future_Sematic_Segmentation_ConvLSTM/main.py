import torch
import argparse

import dataset
import trainer

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# argparser generation
parser = argparse.ArgumentParser()
parser.add_argument("-max_epoch", default=20, help= "sum of epoches")
parser.add_argument("-lr", default=1e-4, type=float, help= "learning rate")
parser.add_argument("-batch_size", default=16, type=int, help= "mini batch")
parser.add_argument("-frames_input", default=10, type=int, help= "sum of input frames")
parser.add_argument("-frames_output", default=1, type=int, help= "sum of predict frames")
parser.add_argument("-save_dir", default="./save_ckpt", type=str, help= "save_model_file")
parser.add_argument("-mode_flag", default="train", type=str, help= "select the mode: [train], [inference]")
args = parser.parse_args()


if __name__ == "__main__":

    train_data, valid_data, test_data = dataset.data_set()

    if args.mode_flag == "train":
        trainer.train(train_data, valid_data, args, device)
    else:
        trainer.inference(test_data, args, device)













