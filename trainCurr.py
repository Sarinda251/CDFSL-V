from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
import numpy as np
import torch
import random
import os
import torchvision
from VideoDataset import FrameSampleDataset
import warnings
from opts import parser
from time import ctime
import json
from itertools import cycle
import torch.nn.functional as F
import math
import copy
import itertools
import torch.nn as nn
from torchvision import transforms
import logging

with open('dataset/datapath.json') as json_file:
    datapath = json.load(json_file)

warnings.simplefilter('ignore')
random.seed(251)

parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--target', default='H51')
parser.add_argument('--temp', default=.1)
args = parser.parse_args()

torchvision.set_video_backend('video_reader')

class FrameSampleDataset2(torch.utils.data.Dataset):
    def __init__(self, annotation, epoch_size=None, frame_transform=None, video_transform=None, clip_len=8, dataset=''):
        super(FrameSampleDataset2).__init__()

        #self.samples = get_samples(root)
        lines = []
        with open(annotation) as f:
            for line in f:
                split = line.strip().split(', ')
                if split[1] == 'label':
                    continue
                if dataset == 'SSm':
                    lines.append((split[0], str(int(split[1]))))
                else:
                    lines.append((split[0], str(int(split[1]) - 1)))

        print('start', lines[0], len(lines))
        self.samples = lines
        self.preprocess = VideoMAEImageProcessor(size= {"height": 224, "width": 224}, do_center_crop=False)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = self.preprocess#frame_transform
        self.video_transform = video_transform

    def __getitem__(self, idx: int):
        # Get random sample
        #path, target = random.choice(self.samples)
        path, target = self.samples[idx]
        # Get video object
#        logger.debug(path)
        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()
        video_frames = []  # video frame buffer
        video_frames2 = []
        image_size = 224
        strongtf = torchvision.transforms.Compose([
            #ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.5, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ])

        weaktf = torchvision.transforms.Compose([
            #ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.5, 1.)),
            #transforms.Resize(112),
            #transforms.CenterCrop(112),
            torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ])

        if not metadata["video"]['duration']:
            print('fail', path)
            #return  {'fail': path}
        # Seek and return frames
        max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
        if metadata["video"]['duration'][0] < 0:
            max_seek = 0
        start = random.uniform(0., max_seek)
        for frame in itertools.islice(vid.seek(start), self.clip_len):
            if self.frame_transform is None:
                video_frames.append(frame['data'])
                video_frames2.append(frame['data'])
            else:
                video_frames.append(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor))
                video_frames2.append(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor))
                #print(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor).size())
            current_pts = frame['pts']
        # Stack it into a tensor
        #print(video_frames)
        #video_frames = self.preprocess(video_frames)
        #print(video_frames)
        if not video_frames:
            print('fail frames ' + path)
        elif len(video_frames) == 0:
            print('zero frames ' + path)
        if not video_frames2:
            print('fail frames ' + path)
        elif len(video_frames2) == 0:
            print('zero frames ' + path)
        while len(video_frames) < self.clip_len:
            video_frames.append(video_frames[-1])
        while len(video_frames2) < self.clip_len:
            video_frames.append(video_frames2[-1])
        video = strongtf(torch.stack(video_frames, 0))
        video2 = weaktf(torch.stack(video_frames2, 0))
        #video = video.type(torch.FloatTensor)
        if self.video_transform:
            video = self.video_transform(video)
            video2 = self.video_transform(video2)
        output = {
            'path': path,
            'video': (video, video2),
            'target': target,
            'start': start,
            'end': current_pts}
        return output
    def __len__(self):
        return self.epoch_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torchvision.set_video_backend('video_reader')
random.seed(251)

def Average(lst):
    return sum(lst) / len(lst)


frame_transform = VideoMAEImageProcessor(size= {"height": 224, "width": 224}, do_center_crop=False)
source_set = FrameSampleDataset(datapath[args.source]['path'], frame_transform=frame_transform)

source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=4, shuffle = True, pin_memory = True, drop_last=True)

target_set = FrameSampleDataset2(datapath[args.target]['path'], frame_transform=frame_transform, clip_len = 8, dataset=args.source)

target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, num_workers=1, shuffle = True, pin_memory = True, drop_last=True)

configuration = VideoMAEConfig()
configuration.num_hidden_layers = 8
configuration.intermediate_size = 2358
configuration.image_size = 224
configuration.num_labels = datapath[args.source]['num_classes']
print("config loaded")

student = VideoMAEForVideoClassification(configuration)
print("model configured")

teacher = copy.deepcopy(student)

if args.load:
    student.load_state_dict(torch.load(args.load), strict=False)
    teacher.load_state_dict(torch.load(args.load.replace("epochS", "epochT")), strict=False)
    print('loaded from: ' + args.load)

teacher.train(True).to(device)
student.train(True).to(device)

student.classifier.requires_grad = False

optimizer = torch.optim.SGD(student.parameters(), lr=.01, momentum=0.9, nesterov=True)

path = "saves/distill_" + args.source + "_" + args.target + "_" + args.run

if not os.path.exists(path):
    os.mkdir(path)

print("start training")

startep = 0
endep = args.max_epochs + 1

criterion = torch.nn.CrossEntropyLoss().to(device)

def adjust_optim(optimizer, n_iter, max_iter):
    ratio = n_iter/max_iter
    optimizer.param_groups[1]['lr'] = ((.005*np.arctan(-100*(ratio-.05)))/(.5 * math.pi)) + .005

def cross_entropy(logits, y_gt):
    if len(y_gt.shape) < len(logits.shape):
        return F.cross_entropy(logits, y_gt, reduction='mean')
    else:
        return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

def distill_loss2(student_out, teacher_out):
    teacher_out /= .1
    loss = cross_entropy(student_out, teacher_out.softmax(dim=-1))
    return loss

def distill_loss(student_out, teacher_out):
    teacher_out /= float(args.temp)
    loss = criterion(student_out, teacher_out.softmax(dim=-1))
    return loss

for epoch in range(startep, endep):
    running_loss = []
    run_loss_S = []
    run_loss_U = []
    #epoch_acc = []
    total = 0
    correct = 0

    target_loader2 = cycle(target_loader)

    for i, (data1, data2) in enumerate(zip(source_loader, target_loader2)):

        video_batch1 = data1['video']
        y = torch.Tensor([int(j) for j in data1['target']]).type(torch.LongTensor).to(device)
        video_batch2 = data2['video']

        (x_u_s, x_u_w) = video_batch2

        x_u_s = torch.nan_to_num(x_u_s, nan = x_u_s.nanmean().item())
        x_u_w = torch.nan_to_num(x_u_w, nan = x_u_w.nanmean().item())

        x_u_s, x_u_w = x_u_s.to(device), x_u_w.to(device)
        optimizer.zero_grad()

        video_batch1 = torch.nan_to_num(video_batch1, nan = video_batch1.nanmean())

        outputs = student(video_batch1.to(device)).logits

        loss_ce = criterion(outputs, y)
        #loss_ce = cross_entropy(outputs, y)

        scores_u = student(x_u_s.to(device)).logits

        #print(scores_u)

        torch.set_grad_enabled(False)
        logit_pseudo = teacher.forward(x_u_w).logits
        torch.set_grad_enabled(True)

        scores_u = torch.nan_to_num(scores_u, nan = scores_u.nanmean().item())
        logit_pseudo = torch.nan_to_num(logit_pseudo, nan = logit_pseudo.nanmean().item())

        loss_pseudo = distill_loss(scores_u, logit_pseudo)
        if math.isnan(loss_pseudo.item()):
            data2['path']
            print(x_u_w.isnan().any())
            print(x_u_w)
            print(scores_u, logit_pseudo)
            exit(0)
        multiplier = ((.05*np.arctan(10*((epoch / args.max_epochs)-.05)))/(.5 * math.pi)) + .05
        multiplier = 1
        adjust_optim(optimizer, epoch, max_epoch)
        loss = loss_ce + (multiplier * loss_pseudo)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        #print(loss.item(), loss_ce.item(), loss_pseudo.item(), multiplier * loss_pseudo)

        running_loss.append(loss.item())
        run_loss_S.append(loss_ce.item())
        run_loss_U.append(loss_pseudo.item())
        #running_loss += loss.item()


    with torch.no_grad():
        m = 0.99  # momentum parameter
        for param_q, param_k in zip(student.parameters(),
                                    teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    print("time: " + ctime() + " epoch" + str(epoch) + ": loss:" + str(Average(running_loss)) + ": lossSup:" + str(Average(run_loss_S)) + ": lossCons:" + str(Average(run_loss_U)) + " acc:" + str(100 * correct / total))

    if epoch%20 == 0 or epoch == 1:
        torch.save(student.state_dict(), path + "/epochS" + str(epoch) + ".pth")
        torch.save(teacher.state_dict(), path + "/epochT" + str(epoch) + ".pth")
