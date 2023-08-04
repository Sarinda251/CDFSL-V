from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
import numpy as np
import torch
import random
import os
import torchvision
from VideoDataset import FrameSampleDataset
import itertools
from torch.optim import SGD
from torch import nn
import torchmetrics
from time import ctime
import warnings
import logging
from opts import parser
import json
import random

parser.add_argument('--max_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

outfile = "preSup_" + args.source + "_B" + str(args.batch_size) + "LrS.01"
if not args.run == 'debug':
    outfile = outfile + "_" + args.run
    
logging.basicConfig(filename=outfile + ".log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug('logged in')

with open('dataset/datapath.json') as json_file:
    datapath = json.load(json_file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torchvision.set_video_backend('video_reader')
random.seed(251)
warnings.simplefilter('ignore')

def Average(lst):
    return sum(lst) / len(lst)

transformsf = [torchvision.transforms.Resize((112, 112)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
#transformsv = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
#frame_transform = torchvision.transforms.Compose(transformsf)
#video_transform = torchvision.transforms.Compose(transformsv)
#source_set = FrameSampleDataset("dataset/ucf101_train_clean.txt", frame_transform=frame_transform)#, video_transform = video_transform)
source_set = FrameSampleDataset(datapath[args.source]['path'], frame_transform=frame_transform, clip_len = args.num_frames, logger = logger)
trainloader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=args.workers, shuffle = True, pin_memory = True, drop_last=True)
logger.debug("data loaded")
    
configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
configuration.num_hidden_layers = 8
configuration.intermediate_size = 2358
configuration.image_size = 112
configuration.num_labels = datapath[args.source]['num_classes']
logger.debug("config loaded")

model = VideoMAEForVideoClassification(configuration)
logger.debug("model configured")

#model.classifier = nn.Linear(768,61)
#model.classifier = nn.Linear(768,101)


model.train(True).to(device)

print("module put on device")
logger.debug("model put on device")

train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=61)
#optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=.05, amsgrad=False)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.9, nesterov=True)

running_loss = []

if args.load:
    model.load_state_dict(torch.load(args.load), strict=False)
    logger.debug('loaded from: ' + args.load)
#model.load_state_dict(torch.load("saves/preSS_K100B16LrS1/epoch1000.pth"), strict=False)

path = "saves/preSup_" + args.source + "_B" + str(args.batch_size) + "LrS.01"
if not args.run == 'debug':
    path = path + "_" + args.run

if not os.path.exists(path):
    os.mkdir(path)

print("start training")
logger.debug("start training")

startep = args.start_ep
endep = args.max_epochs + 1

criterion = torch.nn.CrossEntropyLoss().to(device)   

for epoch in range(startep, endep):
    running_loss = []
    epoch_acc = []
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        imgs = batch['video']
        #print(batch['target'])
        targets = torch.Tensor([int(j) for j in batch['target']]).type(torch.LongTensor).to(device)

        
        outputs = model(imgs.type(torch.FloatTensor).to(device)).logits
        #output = outputs.logits
        loss = criterion(outputs, targets)
        running_loss.append(loss.item())
        
        acc = train_acc(outputs.argmax(dim=-1).cpu(), targets.cpu())
        epoch_acc.append(acc.item())
        
        #print(loss)
        loss.backward()
        optimizer.step()
    print("time: " + ctime() + " epoch" + str(epoch) + ": loss:" + str(Average(running_loss)) + " acc:" + str(Average(epoch_acc)))
        
    if epoch%50 == 0 or epoch == 1:
        torch.save(model.state_dict(), path + "/epoch" + str(epoch) + ".pth")