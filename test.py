import torch
import cv2
import torch
from config import MyConfigs
from datasets import *
from utils.utils import accuracy

def evaluate(test_loader,model,criterion):
    sum = 0
    test_loss_sum = 0
    test_top1_sum = 0
    model.eval()

    for img, label in tqdm(test_loader):
        target_test = torch.tensor(label).cuda()
        y = model(img.cuda())
        loss = criterion(y, target_test)
        test_loss_sum += loss.cpu().item()
        top1_test = accuracy(y, target_test, topk=(1,))
        test_top1_sum += top1_test[0][0].cpu().item()
        sum += 1
    avg_loss = test_loss_sum / sum
    avg_top1 = test_top1_sum / sum
    return avg_loss, avg_top1

def test_one_image(image,model):
    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (MyConfigs.img_width, MyConfigs.img_height))
    image = torch.tensor(image).permute(2,0,1).unsqueeze(0).cuda() / 255.0
    y = model(image).argmax().item()
    return MyConfigs.classes[y]
