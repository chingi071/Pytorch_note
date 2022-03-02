import os
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def train_epoch(train_loader, optimizer, criterion, lr_scheduler, model, world_size):
        model.train()

        train_running_loss = 0.0
        train_running_acc = 0.0    
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)        
            preds = torch.max(output, 1)[1]

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_running_acc += torch.eq(preds, target).sum().item()

        lr_scheduler.step()
        train_loss_value = train_running_loss/ (len(train_dataset) / world_size)
        train_acc_value = train_running_acc/ (len(train_dataset) / world_size)

        return train_loss_value, train_acc_value

def valid_epoch(valid_loader, criterion, model, world_size):
    model.eval()

    valid_running_loss = 0.0
    valid_running_acc = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(data)
            preds = torch.max(outputs, 1)[1]

            loss = criterion(outputs, target)

            valid_running_loss += loss.item() 
            valid_running_acc += torch.eq(preds, target).sum().item()

    valid_loss_value = valid_running_loss/ (len(valid_dataset) / world_size)
    valid_acc_value = valid_running_acc/ (len(valid_dataset) / world_size)

    return valid_loss_value, valid_acc_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    dist.barrier()
#     rank = dist.get_rank()
    world_size = dist.get_world_size()


    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=256,
                              pin_memory=False, prefetch_factor=2, num_workers=4)

    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=256,
                              pin_memory=False, prefetch_factor=2, num_workers=4)

    if torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=128), nn.LeakyReLU(),
                             nn.Dropout(0.5), nn.Linear(128, 10))

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    lr_scheduler_values = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        train_loss_value, train_acc_value = train_epoch(train_loader, optimizer, criterion, lr_scheduler_values, model, world_size)

        valid_loss_value, valid_acc_value = valid_epoch(valid_loader, criterion, model, world_size)    

        print("Train_local_rank: {} Train_Epoch: {}/{} Training_Loss: {} Training_acc: {:.2f}\
                   ".format(args.local_rank, epoch, num_epochs-1, train_loss_value, train_acc_value))

        print("Valid_local_rank: {} Valid_Epoch: {}/{} Valid_Loss: {} Valid_acc: {:.2f}\
                   ".format(args.local_rank, epoch, num_epochs-1, valid_loss_value, valid_acc_value))

        print('--------------------------------')

    print("finished.")