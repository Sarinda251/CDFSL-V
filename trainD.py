from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
import math
from opts import parser
import numpy as np
import torch
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
global args, best_prec1, writer
import time
import os
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import torchvision
import torchmetrics
from itertools import cycle
import torch.nn.functional as F
import math
import copy
import random
import itertools
from time import ctime
import warnings
random.seed(251)
import logging
parser.add_argument('--source', default='K100')
parser.add_argument('--target', default='H51')
parser.add_argument('--max_epochs', default=150)
args = parser.parse_args()
import json
from VideoDataset import FrameSampleDataset

with open('dataset/datapath.json') as json_file:
    datapath = json.load(json_file)

outfile = "curriculum_" + args.source + "_" +args.target
if not args.run == 'debug':
    outfile = outfile + "_" + args.run
    
logging.basicConfig(filename=outfile + ".log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug('logged in')

warnings.simplefilter('ignore')
torchvision.set_video_backend('video_reader')

print(torchvision.get_video_backend())

#torchvision.set_video_backend("video_reader")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class MultiTransform(object):
    def __init__(self, transforms: list):
        self.base_transforms = transforms

    def __call__(self, x):
        out = []
        for tfm in self.base_transforms:
            out.append(tfm(x))
        return out
        


#targettf = MultiTransform([strongtf, weaktf])

class FrameSampleDataset2(torch.utils.data.Dataset):
    def __init__(self, annotation, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(FrameSampleDataset2).__init__()

        #self.samples = get_samples(root)
        lines = []
        with open(annotation) as f:
            for line in f:
                split = line.strip().split(', ')
                if split[1] == 'label':
                    continue
                lines.append((split[0], str(int(split[1]) - 1)))
        
        print('start', lines[0], len(lines))
        self.samples = lines
        self.preprocess = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)

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
        
        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()
        video_frames = []  # video frame buffer
        video_frames2 = []
        
        strongtf = torchvision.transforms.Compose([
            #ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            #transforms.Resize(112),
            #transforms.CenterCrop(112),
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.5, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            #torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])],p=0.5),
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
                video_frames.append(strongtf(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor)))
                video_frames2.append(weaktf(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor)))
                #print(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor).size())
            current_pts = frame['pts']
        # Stack it into a tensor
        #print(video_frames)
        #video_frames = self.preprocess(video_frames)
        #print(video_frames)
        if not video_frames:
            print('fail frames ' + path)
        while len(video_frames) < self.clip_len:
            video_frames.append(video_frames[-1])
        while len(video_frames2) < self.clip_len:
            video_frames.append(video_frames2[-1])
        video = torch.stack(video_frames, 0)
        video2 = torch.stack(video_frames2, 0)
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
        
print(os.getcwd())

image_size = 112

frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
source_set = FrameSampleDataset(datapath[args.source]['path'], frame_transform=frame_transform)
    
source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=4, shuffle = True, pin_memory = True, drop_last=True)
                                                                                                         
target_set = FrameSampleDataset2(datapath[args.target]['path'], frame_transform=frame_transform)
    
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, num_workers=4, shuffle = True, pin_memory = True, drop_last=True)

#for i, data in enumerate(target_loader):
#    print(data['target'])

ckpt = "saves/pretrain_K_R2+1D_official/r2plus1d_18-91a641e6.pth"

model = []

num_frames = 16

configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
configuration.num_hidden_layers = 8
configuration.intermediate_size = 2358
configuration.image_size = 112
print("config loaded")
logger.debug("config loaded")

student = VideoMAEForVideoClassification(configuration)
print("model configured")
logger.debug("model configed")

student.classifier = nn.Linear(768,61)
#student.classifier.requires_grad = False
#student = model
#print(student)
model = []
teacher = copy.deepcopy(student)                          
          
student.load_state_dict(torch.load(args.load), strict=False)
teacher.load_state_dict(torch.load(args.load), strict=False)


#student = nn.DataParallel(student)
#teacher = nn.DataParallel(teacher)    
student = student.to(device)
teacher = teacher.to(device)


criterion = torch.nn.CrossEntropyLoss().to(device)   

optimizer = torch.optim.SGD(student.parameters(), 0.0001, momentum=0.9, weight_decay=1e-4, nesterov=False)
#print(optimizer.param_groups)
#optimizer = torch.optim.SGD([{'params': student.videomae.parameters()}, {'params': student.classifier.parameters()}], lr=.001, momentum=0.9, weight_decay=1e-4, nesterov=False)

#print('hi', optimizer.param_groups)

def adjust_optim(optimizer, n_iter, max_iter):
    ratio = n_iter/max_iter
    optimizer.param_groups[1]['lr'] = ((.005*np.arctan(-500*(ratio-.05)))/(.5 * math.pi)) + .005

train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=61)

def Average(lst):
    return sum(lst) / len(lst)

start_epoch = 0
start_train = time.time()

#for videos, labels in source_loader:
#    print(videos.size())

if not os.path.exists("saves/distill" +  args.source + '_' + args.target):
    os.mkdir("saves/distill" +  args.source + '_' + args.target)
print('start training......')
logger.debug('start training')
max_epoch = 301

for epoch in range(start_epoch, max_epoch):
#for epoch in range(start_epoch, args.max_epochs+1):
    running_loss = []
    run_loss_S = []
    run_loss_U = []
    epoch_acc = []
    
    #dataloader_iterator = iter(target_loader)
    
    #for i, data1 in enumerate(source_loader):

    #    try:
    #        data2 = next(dataloader_iterator)
    #    except StopIteration:
    #        dataloader_iterator = iter(dataloaders1)
    #        data2 = next(dataloader_iterator)
            
            
    for i, (data1, data2) in enumerate(zip(source_loader, target_loader)):
    #for i, (data1, data2) in enumerate(zip(cycle(source_loader), target_loader)):
    #for video_batch, labels in source_loader:
        #video_batch1, y = data1
        video_batch1 = data1['video']
        y = torch.Tensor([int(j) for j in data1['target']]).type(torch.LongTensor).to(device)

        #print(video_batch1, y)
        video_batch2 = data2['video']
        
        (x_u_s, x_u_w) = video_batch2
        #print(video_batch2.shape)
        #x_u_s = strongtf(video_batch2)
        #x_u_w = weaktf(video_batch2)
        #print(video_batch1.shape, x_u_s.shape)
        #print(x_u_s.shape)
        
        x_u_s, x_u_w = x_u_s.to(device), x_u_w.to(device)
        optimizer.zero_grad()
        
        outputs = student(video_batch1.to(device)).logits
        
        #print(outputs.shape)
        #print(y.shape)
        loss_ce = criterion(outputs, y)
        #print('HOLYSHITITWORKS')
        
        scores_u = student(x_u_s.to(device)).logits
        
        #scores_u = student.forward(x_u_s.permute(0, 2, 1, 3, 4))
        
        torch.set_grad_enabled(False)
        logit_pseudo = teacher.forward(x_u_w).logits
        #logit_pseudo = teacher.forward(x_u_w.permute(0, 2, 1, 3, 4))
        torch.set_grad_enabled(True)
        
        def cross_entropy(logits, y_gt):
            if len(y_gt.shape) < len(logits.shape):
                return F.cross_entropy(logits, y_gt, reduction='mean')
            else:
                return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()       
        def distill_loss(student_out, teacher_out):
            teacher_out /= .1
            loss = cross_entropy(student_out, teacher_out.softmax(dim=-1))
            return loss

                
        loss_pseudo = distill_loss(scores_u, logit_pseudo)
        if (epoch / max_epoch) < .5:
            multiplier = 0
        else:
            multiplier = min(1, 0.75 * (1 - math.cos(math.pi * epoch / max_epoch)))
        
        #multiplier = min(1, 0.75 * (1 - math.cos(math.pi * epoch / max_epoch)))    
        #print("mult", multiplier)
        #multiplier = min(1, (1 - math.cos(math.pi * epoch / max_epoch)
        
        #print("class lr", str(((.05*np.arctan(-500*((epoch / max_epoch)-.05)))/(.5 * math.pi)) + .05))
        #adjust_optim(optimizer, epoch, max_epoch)
        #multiplier = (1 - math.cos(.5 * math.pi * epoch / 151))
        loss = loss_ce + (multiplier * loss_pseudo)
        loss.backward()
        optimizer.step()
        
        acc = train_acc(outputs.argmax(dim=-1).cpu(), y.cpu())
        epoch_acc.append(acc.item())
        running_loss.append(loss.item())
        run_loss_S.append(loss_ce.item())
        run_loss_U.append(loss_pseudo.item())
        #running_loss += loss.item() 
        
        
    with torch.no_grad():
        m = 0.99  # momentum parameter
        for param_q, param_k in zip(student.parameters(),
                                    teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    print("time: " + ctime() + "epoch" + str(epoch) + ": loss:" + str(Average(running_loss)) + ": loss_S:" + str(Average(run_loss_S)) + ": loss_U:" + str(Average(run_loss_U)) + ": source acc:" + str(Average(epoch_acc)))
    logger.debug("time: " + ctime() + "epoch" + str(epoch) + ": loss:" + str(Average(running_loss)) + ": loss_S:" + str(Average(run_loss_S)) + ": loss_U:" + str(Average(run_loss_U)) + ": source acc:" + str(Average(epoch_acc)))
    #print(running_loss)
    #print(acc.mean())
            
    if epoch%10 == 0 or epoch == 1:
        torch.save(student.state_dict(), "saves/distill" +  args.source + '_' + args.target + "/Sepoch" + str(epoch) + ".pth")
        torch.save(teacher.state_dict(), "saves/distill" +  args.source + '_' + args.target + "/Tepoch" + str(epoch) + ".pth")
            