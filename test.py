import torch
import cv2
import torch
from torch import nn
from torch.autograd import Variable
from config import MyConfigs
from datasets import *
from utils.utils import accuracy

def evaluate(test_loader,model,criterion):
    sum = 0
    test_loss_sum = 0
    test_top1_sum = 0
    model.eval()

    for img, label in test_loader:
        target_test = torch.tensor(label).cuda()
        output_test = model(img.cuda())
        loss = criterion(output_test, target_test)
        top1_test = accuracy(output_test, target_test, topk=(1,))
        sum += 1
        test_loss_sum += loss.data.cpu().numpy()
        test_top1_sum += top1_test[0].cpu().numpy()[0]
    avg_loss = test_loss_sum / sum
    avg_top1 = test_top1_sum / sum
    return avg_loss, avg_top1

def test(test_loader,model):
    model.eval()
    predict_file = open("%s.txt" % MyConfigs.model_name, 'w')
    for input,filename in tqdm(test_loader):
        y_pred = model(input).argmax().item()
        predict_file.write(filename[0]+', ' + MyConfigs.classes[y_pred]+'\n')

def test_one_image(image,model):
    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (MyConfigs.img_width, MyConfigs.img_height))
    image = torch.tensor(image).permute(2,0,1).unsqueeze(0).cuda() / 255.0
    y = model(image).argmax().item()
    return MyConfigs.classes[y]
