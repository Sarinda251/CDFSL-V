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
import logging
warnings.simplefilter('ignore')

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

def Average(lst):
    return sum(lst) / len(lst)

frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
source_set = FrameSampleDataset(datapath[args.source]['path'], frame_transform=frame_transform, clip_len = args.num_frames, logger = logger)
trainloader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=4, shuffle = True, pin_memory = True, drop_last=True)

configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
configuration.num_hidden_layers = 8
configuration.intermediate_size = 2358
configuration.image_size = 112
configuration.num_labels = datapath[args.source]['num_classes']
print("config loaded")

model = VideoMAEForVideoClassification(configuration)
print("model configured")

if args.load:
    model.load_state_dict(torch.load(args.load), strict=False)
    print('loaded from: ' + args.load)

model.train(True).to(device)

#train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=61)
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.9, nesterov=True)

path = "saves/testSup_U101" + "_B" + str(32) + "LrS.01" + args.run

if not os.path.exists(path):
    os.mkdir(path)
    
print("start training")

startep = 0
endep = args.max_epochs + 1

criterion = torch.nn.CrossEntropyLoss().to(device)   

for epoch in range(startep, endep):
    running_loss = []
    #epoch_acc = []
    total = 0
    correct = 0
    for i, batch in enumerate(trainloader):
        imgs = batch['video']
        targets = torch.Tensor([int(j) for j in batch['target']]).type(torch.LongTensor).to(device)
        optimizer.zero_grad()

        
        outputs = model(imgs.type(torch.FloatTensor).to(device)).logits
        loss = criterion(outputs, targets)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        #acc = train_acc(outputs.argmax(dim=-1).cpu(), targets.cpu())
        #epoch_acc.append(100 * correct // total)
        
        #print(loss)
    print("time: " + ctime() + " epoch" + str(epoch) + ": loss:" + str(Average(running_loss)) + " acc:" + str(100 * correct / total))
        
    if epoch%50 == 0 or epoch == 1:
        torch.save(model.state_dict(), path + "/epoch" + str(epoch) + ".pth")