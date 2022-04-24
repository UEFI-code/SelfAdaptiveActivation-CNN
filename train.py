import numpy as np
import torch
import torchvision
import os
from config import config
import Model
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import *
from test import *
from utils.utils import*
if __name__ == '__main__' :
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    model = Model.myModel()
    if torch.cuda.is_available():
        model =model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    current_accuracy = 0
    resume = False
    if resume:
        checkpoint = torch.load(config.weights+ config.model_name+'.pth')
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    transform = transforms.Compose([
                                    transforms.RandomResizedCrop(90),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(30),
                                    transforms.RandomGrayscale(p = 0.5),
                                    transforms.Resize((config.img_width, config.img_height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    _, train_list = get_files(config.data_folder, config.ratio)
    input_data = datasets(train_list, transform = transform)
    train_loader = DataLoader(input_data, batch_size = config.batch_size, shuffle = True, collate_fn = collate_fn, pin_memory = False, num_workers = config.num_worker)
    test_list, _ = get_files(config.data_folder, config.ratio)
    test_loader = DataLoader(datasets(test_list, transform = None), batch_size= config.batch_size, shuffle = False, collate_fn = collate_fn, num_workers = config.num_worker)
    train_loss = []
    acc = []
    test_loss = []
    print("------ Start Training ------\n")
    for epoch in range(start_epoch,config.epochs):
        model.train()
        config.lr = lr_step(epoch)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)

        loss_epoch = 0
        for index,(input,target) in enumerate(train_loader):
            model.train()
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(input)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if (index+1) % 10 == 0:
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch+1,index*config.batch_size,len(train_loader.dataset),loss.item()))
        if (epoch+1) % 1 ==0:
            print("\n------ Evaluate ------")
            model.eval()
            test_loss1, accTop1 = evaluate(test_loader,model,criterion)
            acc.append(accTop1)
            print("type(accTop1) =",type(accTop1))
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch/len(train_loader))
            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch+1,accTop1,test_loss1))
            save_model = accTop1 > current_accuracy
            accTop1 = max(current_accuracy,accTop1)
            current_accuracy = accTop1
            print("Best Accu = %f" % current_accuracy)
            save_checkpoint({
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, save_model)

