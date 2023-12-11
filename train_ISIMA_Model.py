import os
import sys

from torch import optim, nn
from torch.utils.data import DataLoader
from datasets.dataset import FakeDataset

import CustemModel

from extractParams import *
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train ISMA Modle')
    parser.add_argument("--train_path", type=str, help="path to the file with input paths") # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument('--val_path_cls', type=str, help='path to the pretrained cls model', default="paths/val_cls.txt")
    parser.add_argument('--val_path_seg', type=str, help='path to the pretrained seg model', default="paths/val_seg.txt")

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--target_part', type=str, help='path to the pretrained seg model', default="None")
    parser.add_argument('--load_path_main', type=str, help='path to the pretrained main model', default="checkPoints/9.pth")
    parser.add_argument('--load_path_cls', type=str, help='path to the pretrained cls model', default="None")
    parser.add_argument('--load_path_seg', type=str, help='path to the pretrained seg model', default="None")

    parser.add_argument("--image_size", type=int, default=512, help="size of the images for prediction")
    parser.add_argument("--batch_size", type=int, help="Batch size for the dataloader" ,default= 16)
    parser.add_argument("--nbr_worker", type=int, help="nomber of threads worker for the dataloader", default= 4)
    parser.add_argument("--learn_rate", type=int, help="learn rate for the model", default= 1e-4)
    parser.add_argument("--nbr_epochs", type=int, help="nomber of epochs for training the model", default= 10)
    parser.add_argument("--full_train", type=bool, help="train the full model layers" , default=False)

    parser.add_argument('--device', type=str, help='device to train the model on cuda / cpu', default="cuda")
    args = parser.parse_args()
    return args

def init_dataset(path, batch_size, nbr_worker, val = False):
    dataset = FakeDataset(0,path,512, 0, None, val)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=nbr_worker, pin_memory=True, shuffle = (not val))

    return dataloader 

def train_cls(loader, model, optimizer, loss_fn, scaler, current_epoch):
    loop = tqdm(loader)
    info_loss = []
    for batch_idx, (in_imgs, _ , _ , in_labels) in enumerate(loop):
        in_imgs = in_imgs.cuda()
        in_labels = in_labels.float().unsqueeze(1).cuda()
        
        #forward prop
        with torch.cuda.amp.autocast():
            prediction , _= model(in_imgs)
            loss = loss_fn(prediction, in_labels)

        #backward prop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        info_loss.append(str(batch_idx) + " " + str(loss.item()))

    with open(f"info/cls_{current_epoch}.txt", "w") as f:
        for row in info_loss:
            f.write("".join(row) + "\n")

def train_seg(loader, model, optimizer,loss_fn, scaler, current_epoch):
    loop = tqdm(loader)
    info_loss = []
    for batch_idx, (in_imgs, in_mask , _ , _) in enumerate(loop):
        in_imgs = in_imgs.cuda()
        in_mask = in_mask.cuda()

        #forward prop
        with torch.cuda.amp.autocast(): #training a5af w asra3
            _ , prediction = model(in_imgs)
            loss = loss_fn(prediction, in_mask)

        #backward prop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        info_loss.append(str(batch_idx) + " " + str(loss.item()))

    with open(f"info/seg_{current_epoch}.txt", "w") as f:
        for row in info_loss:
            f.write("".join(row) + "\n")
    
def main():
    arg = parse_args()

    train_target = arg.target_part

    cls_pretraint = None
    seg_pretraint = None
    if arg.load_path_cls != "None":
        train_target = "cls"
        cls_pretraint = torch.load(arg.load_path_cls)
    if arg.load_path_seg != "None":
        train_target = "seg"
        seg_pretraint = torch.load(arg.load_path_seg)
    if train_target == "None":
        print("must train one part at a time")
        sys.exit()
    
    model = load_paper_model("ours", arg.load_path_main, arg.image_size).cuda()

    for param in model.parameters():
        param.requires_grad = False
    
    print(f"setup {train_target} for the training")
    model.decoder = CustemModel.CusstemModule(main_model=model, 
                                              train_part=train_target, 
                                              cls_head=cls_pretraint, 
                                              seg_head=seg_pretraint, 
                                              train_main=arg.full_train).cuda()
    
    print(f"setup training params")
    optimizer = optim.Adam(model.parameters(),arg.learn_rate)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"geting the taining data")
    train_loader = init_dataset(arg.train_path, arg.batch_size, arg.nbr_worker)

    print(f"Start")
    for epoch in range(arg.nbr_epochs):
        if train_target == "cls":
            train_cls(train_loader, model, optimizer,loss_fn, scaler, epoch)
            torch.save(model.decoder.cls_head, f"checkPoints/cls_head/{epoch}sec_model.pth")

        elif train_target == "seg":
            train_seg(train_loader, model, optimizer,loss_fn, scaler, epoch)
            torch.save(model.decoder.seg_UNet, f"checkPoints/seg_head/{epoch}.pth")
        

if __name__ == '__main__':
    main()