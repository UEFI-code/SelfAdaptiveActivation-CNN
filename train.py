import torch
import os
from config import MyConfigs
import Model
from torch import optim,nn
from torch.utils.data import DataLoader
from datasets import *
from test import *
from utils.utils import *

if __name__ == '__main__' :
    if not os.path.exists(MyConfigs.weights):
        os.mkdir(MyConfigs.weights)
    if not os.path.exists(MyConfigs.logs):
        os.mkdir(MyConfigs.logs)
    model = Model.myModel()
    if torch.cuda.is_available():
        model =model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=MyConfigs.lr, momentum=0.9, weight_decay=MyConfigs.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    current_accuracy = 0
    resume = os.path.exists(MyConfigs.weights + MyConfigs.model_name+'.pth')
    if resume:
        checkpoint = torch.load(MyConfigs.weights + MyConfigs.model_name+'.pth')
        start_epoch = checkpoint["epoch"]
        current_accuracy = checkpoint["accTop1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    transform = transforms.Compose([
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(30),
                                    transforms.RandomGrayscale(p = 0.5),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),])

    _, train_list = get_files(MyConfigs.data_folder, MyConfigs.ratio)
    train_loader = DataLoader(datasets(train_list, transform = transform, bigRAM=False), batch_size = MyConfigs.batch_size, shuffle = True, pin_memory = False, num_workers = MyConfigs.num_worker)
    test_list, _ = get_files(MyConfigs.data_folder, MyConfigs.ratio)
    test_loader = DataLoader(datasets(test_list, transform = None, bigRAM=False), batch_size= MyConfigs.batch_size, shuffle = False, num_workers = MyConfigs.num_worker)
    train_loss = []
    acc = []
    test_loss = []
    print("------ Start Training ------\n")
    for epoch in range(start_epoch,MyConfigs.epochs):
        model.train()
        MyConfigs.lr = lr_step(epoch)
        optimizer = optim.SGD(model.parameters(), lr=MyConfigs.lr, momentum=0.9, weight_decay=MyConfigs.weight_decay)

        loss_epoch = 0
        for index,(input,target) in enumerate(train_loader):
            model.train()
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if (index+1) % 10 == 0:
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch+1,index*MyConfigs.batch_size,len(train_loader.dataset),loss_epoch/index))
        if (epoch+1) % 1 ==0:
            print("\n------ Evaluate ------")
            model.eval()
            test_loss1, accTop1 = evaluate(test_loader,model,criterion)
            acc.append(accTop1)
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch/len(train_loader))
            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch+1,accTop1,test_loss1))
            save_model = accTop1 > current_accuracy
            save_checkpoint({
                "epoch": epoch + 1,
                "model_name": MyConfigs.model_name,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, save_model)
            if save_model:
                current_accuracy = accTop1
