import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.load_data import get_path
from data.dataset import CustomDataset

from models.Model import Model
from models.Unet3Du import UNet3D
from train.trainer import train_loop

from utils.parser import set_parser
from utils.loss import *

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    
    # load data
    get_path(args)
 
    train_CT = pd.read_excel(f"F:\\HIPPO\\train.xlsx")['CT']
    train_HIPPO = pd.read_excel(f"F:\\HIPPO\\train.xlsx")['HIPPO']

    print(f"Train Set : {len(train_CT)}")

    # make dataset
    train_dataset = CustomDataset(args, train_CT, train_HIPPO)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=args.shuffle)

    # Define Model
    model = UNet3D(1, 1).to(device)
    model = nn.DataParallel(model).to(device)
    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    loss_fn = BCEDiceLoss(1, 1).to(device) # bcedice loss

    save_path = f"F:\\HIPPO\\{args.date}\\model_parameters"
    filename = f"{args.model}_seed.pt"
    model_save_path = os.path.join(save_path, filename)

    # training
    total_loss = train_loop(args, train_dataloader, model, optimizer, loss_fn, model_save_path)

    plt.title(f'{args.model}')
    plt.plot(total_loss)
    plt.savefig(f"./{filename.split('.')[0]}.png")

if __name__=='__main__':
    args = set_parser()
    main(args)