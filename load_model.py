import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from tkinter import Variable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import json
from torch.utils.tensorboard import SummaryWriter
from resunetpluspluscopy import ResUnetPlusPlus

from dice_losscopy import dice_loss

temp_dir = 'E:/data_endovis17/tmp'

val_losses = []
time_taken = []
use_gpu = torch.cuda.is_available()
writer = SummaryWriter(temp_dir)

def disentangleKey(key):

    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = color_array

    return dKey

def convertToOneHot(batch, use_gpu):

    if use_gpu:
        batch = batch.cpu()
    
    batch = batch.data.numpy()
    for i in range(len(batch)):
        vec = batch[i,:,:,:]
        idxs = np.argmax(vec, axis=0)

        single = np.zeros([1, batch.shape[2], batch.shape[3]])
        
        for k in range(batch.shape[1]):
            mask = idxs == k
            mask = np.expand_dims(mask, axis=0)
            single = np.concatenate((single, mask), axis=0)

        single = np.expand_dims(single[1:,:,:], axis=0)
        if 'oneHot' in locals():
            oneHot = np.concatenate((oneHot, single), axis=0)
        else:
            oneHot = single

    oneHot = torch.from_numpy(oneHot.astype(np.uint8))
    return oneHot

def normalize(batch, mean, std):
    
    mean.unsqueeze_(1).unsqueeze_(1)
    std.unsqueeze_(1).unsqueeze_(1)
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = img.sub(mean).div(std).unsqueeze(0)

        if 'concat' in locals():
            concat = torch.cat((concat, img), 0)
        else:
            concat = img

    return concat

def generateOneHot(gt, key):

    batch = gt.numpy()

    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        for k in range(len(key)):
            catMask = catMask * 0
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            catMask[mask] = 1

            catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
            if 'oneHot' in locals():
                oneHot = torch.cat((oneHot, catMaskTensor), 0)
            else:
                oneHot = catMaskTensor

    label = oneHot.view(len(batch),len(key),img.shape[0],img.shape[1])
    return label

def displaySamples(img, generated, gt, use_gpu, key, save, epoch, imageNum,
    save_dir):
    
    if use_gpu:
        img = img.cpu()
        generated = generated.cpu()

    gt = gt.numpy()
    gt = np.transpose((gt[0,:,:,:]), (1,2,0))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    generated = generated.data.numpy()
    generated = reverseOneHot(generated, key)
    
    generated = (np.squeeze(generated[0,:,:,:])).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255.0

    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0,:,:,:]), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stacked = np.concatenate((img, generated, gt), axis = 1)

    if save:
        file_name = 'epoch_%d_img_%d.png' %(epoch, imageNum)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, stacked*255.0)

    cv2.namedWindow('Input | Gen | GT', cv2.WINDOW_NORMAL)
    cv2.imshow('Input | Gen | GT', stacked)

    cv2.waitKey(1)

def reverseOneHot(batch, key):

    for i in range(len(batch)):
        vec = batch[i,:,:,:]
        idxs = np.argmax(vec, axis=0)

        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k

            segSingle[mask] = rgb

        segMask = np.expand_dims(segSingle, axis=0)
        if 'generated' in locals():
            generated = np.concatenate((generated, segMask), axis=0)
        else:
            generated = segMask

    return generated

def generateLabel4CE(gt, key):

    batch = gt.numpy()
    
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = 2))
            catMask[mask] = k

        catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
        if 'label' in locals():
            label = torch.cat((label, catMaskTensor), 0)
        else:
            label = catMaskTensor

    return label.long()

class Evaluate():
    
    def __init__(self, num_classes, use_gpu):
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def addBatch(self, seg, gt):
        seg = convertToOneHot(seg, self.use_gpu).byte()
        seg = seg.float()
        gt = gt.float()
    
        if not self.use_gpu:
            seg = seg.cuda()
            gt = gt.cuda()
        
        tpmult = seg * gt
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim = 0, keepdim = True), dim = 2, keepdim = True), dim = 3, keepdim = True).squeeze()
        fpmult = seg * (1-gt)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-seg) * (gt)
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return iou
    
    def getPRF1(self):
        precision = self.tp / (self.tp + self.fp + 1e-15)
        recall = self.tp / (self.tp + self.fn + 1e-15)
        f1 = (2 * precision * recall) / (precision + recall + 1e-15)

        return precision, recall, f1

class MyDataset(Dataset):

    def __init__(self, root_dir, transform, json_path):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'selfimages')
        self.gt_dir = os.path.join(root_dir, 'selfgroundtruth1')
        self.image_list = [f for f in os.listdir(self.img_dir) if (f.endswith('.png'))]
        self.transform = transform

        if json_path:
            self.classes = json.load(open(json_path))['classes']

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        gt_name = os.path.join(self.gt_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        gt = Image.open(gt_name)

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
        
        return image, gt



def main():
    
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((256, 256), interpolation = Image.NEAREST),
            transforms.ToTensor()
        ]),
        'test' : transforms.Compose([
            transforms.Resize((256, 256), interpolation = Image.NEAREST),
            transforms.ToTensor()
        ]),
    }

    data_dir = 'E:/data_endovis17/selfbuilder'
    json_path = 'E:/data_endovis17/endovis2017Classes.json'

    image_datasets = {x: MyDataset(os.path.join(data_dir, x), data_transforms[x], json_path) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle = True, num_workers = 4) for x in ['train', 'test']}

    classes = image_datasets['train'].classes
    key = disentangleKey(classes)
    num_classes = len(key)

    model = ResUnetPlusPlus(channel = 3)

    if use_gpu:
        model.cuda()
    
    evaluator = Evaluate(key, use_gpu)
    
    for epoch in range(0, 50):
        

        print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
        validate(dataloaders['test'], model, epoch, key, evaluator)
        
        

        print('>>>>>>>>>>>>>>>>>> Evaluating the Metrics <<<<<<<<<<<<<<<<<')
        IoU = evaluator.getIoU()
        print('Mean IoU: {}, Class-wise IoU: {}'.format(torch.mean(torch.tensor(IoU)), IoU))
        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]
        print('Mean Precision: {}, Class-wise Precision: {}'.format(torch.mean(precision), precision))
        print('Mean Recall: {}, Class-wise Recall: {}'.format(torch.mean(recall), recall))
        print('Mean F1: {}, Class-wise F1: {}'.format(torch.mean(F1), F1))
        evaluator.reset()
        
        writer.add_scalar('IoU', torch.mean(torch.tensor(IoU)), epoch)
        writer.add_scalar('Pre', torch.mean(precision), epoch)
        writer.add_scalar('Rec', torch.mean(recall), epoch)
        writer.add_scalar('F1', torch.mean(F1), epoch)

    writer.close()

def validate(val_loader, model, epoch, key, evaluator):
    
    model.eval()
    val_loss = 0

    for i, (img, gt) in enumerate(val_loader):

        
        gt_temp = gt * 255
        label = generateLabel4CE(gt_temp, key)
        oneHotGT = generateOneHot(gt_temp, key)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            
            start_time = time.time()
            
            seg = model(img)
            
            end_time = time.time() - start_time
            time_taken.append(end_time)
            print("{:.10f}".format(end_time))
        
        loss = dice_loss(seg, label)
        val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, 50-1, i, len(val_loader)-1, loss.mean()))
        writer.add_scalar('validateloss', loss.mean(), epoch*len(val_loader)+i)
        displaySamples(img, seg, gt, use_gpu, key, True, epoch,
                             i, temp_dir)
        evaluator.addBatch(seg, oneHotGT)
    
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)
    writer.add_scalar('FPS', mean_fps, epoch)
