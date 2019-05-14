import train_utils as ut

import torch
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help="Directory with data for training, must have 'train' and 'valid' sub directories")
parser.add_argument('--save_dir', help="Directory to save model")
parser.add_argument('--arch', help="Architecture, choicesa are vgg16 or densenet161.", 
                    choices=['vgg16', 'densenet161'], default='vgg16')
parser.add_argument('--lr', help="Learning rate", type=float, default=0.01)
parser.add_argument('--hidden_units', help="Hidden units in the layers, three required.",
                    default=[1024,512,256], type=int, nargs=3)
parser.add_argument('--epochs', help="Number of training epochs", type=int, default=10)
parser.add_argument('--gpu', help="Flag to use GPU for training data, recommended if GPU available.",
                    action='store_true')


A = parser.parse_args()
print('Inputs:')
print(A)
if A.save_dir is None:
    save_dir = os.getcwd()
else:
    save_dir = A.save_dir
print(save_dir)
print("Making data loaders")
dataloaders, class_to_idx = ut.make_data_loaders(A.data_dir)

print("Making model")
model = ut.make_model(A.arch, A.hidden_units)

print("Training model")
model, optimizer, e, running_loss = ut.train_model(model, dataloaders, A.lr, A.epochs, A.gpu)

print("Saving Model")
model.class_to_idx = class_to_idx

if os.path.isdir(save_dir):
    os.chdir(save_dir)
else:
    os.mkdir(save_dir)
    os.chdir(save_dir)
torch.save({'model_state': model.state_dict(),
            'arch': A.arch,
            'hidden_units': A.hidden_units,
            'optimizer_state': optimizer.state_dict(),
            'epoch': e,
            'loss': running_loss,
            'class_to_idx': model.class_to_idx}, 'Model.pt')

print("Done")