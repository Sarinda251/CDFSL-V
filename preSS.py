from transformers import VideoMAEForPreTraining, VideoMAEConfig, VideoMAEImageProcessor
import numpy as np
import torch
import random
import os
import torchvision
import itertools
from torch.optim import SGD
from torch import nn
from opts import parser
from time import ctime
from VideoDataset import FrameSampleDataset
import warnings
import logging
import json

parser.add_argument('--max_epochs', type=int, default=1600)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

outfile = "pre_" + args.source + "_B" + str(args.batch_size) + "LrS.1"
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
        
#transformsf = [torchvision.transforms.Resize((112, 112)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
#transformsv = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
#frame_transform = torchvision.transforms.Compose(transformsf)
frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
#video_transform = torchvision.transforms.Compose(transformsv)
#source_set = FrameSampleDataset("dataset/ucf101_train_clean.txt", frame_transform=frame_transform)
#source_set = FrameSampleDataset("dataset/kinetics100_train_cut_clean.txt", frame_transform=frame_transform)
source_set = FrameSampleDataset(datapath[args.source]['path'], frame_transform=frame_transform, clip_len = args.num_frames, logger = logger)
trainloader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=args.workers, shuffle = True, pin_memory = True, drop_last=True)
logger.debug("data loaded")
    
configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
configuration.num_hidden_layers = 8
configuration.intermediate_size = 2358
configuration.image_size = 112
logger.debug("config loaded")

model = VideoMAEForPreTraining(configuration)
logger.debug("model configured")

#processor = VideoMAEFeatureExtractor(size= {"height": 112, "width": 112}, do_center_crop=False)

num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
seq_length = (args.num_frames // model.config.tubelet_size) * num_patches_per_frame
#model = nn.DataParallel(model).to(device)
model.to(device)

logger.debug("module put on device")

#optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=.05, amsgrad=False)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9, nesterov=True)

running_loss = []

if args.load:
    model.load_state_dict(torch.load(args.load))
    logger.debug('loaded from: ' + args.load)

path = "saves/pre_" + args.source + "_B" + str(args.batch_size) + "LrS.1"
#path = "saves/preSS_U101B16LrS1"

if not os.path.exists(path):
    os.mkdir(path)

logger.debug("start training")

startep = args.start_ep
endep = args.max_epochs + 1

for epoch in range(startep, endep):
    running_loss = []
    for i, batch in enumerate(trainloader):
        optimizer.zero_grad()
        imgs = batch['video']
        #imgs = processor(imgs)
        #print(imgs.shape)
        
        bool_masked_pos = torch.randint(0, 2, (1, seq_length)).repeat(args.batch_size,1).bool()
        #print(imgs.shape[1])
        if not (imgs.shape[1] == args.batch_size):
            logger.error("fail: " + str(batch['path']))
            continue
        
        outputs = model(imgs.type(torch.FloatTensor).to(device), bool_masked_pos=bool_masked_pos)
        loss = outputs.loss
        running_loss.append(loss.item())
        #print(loss)
        loss.backward()
        optimizer.step()
    logger.debug("epoch" + str(epoch) + ": loss:" + str(Average(running_loss)))
    #print(str(1/0))    
    if epoch%25 == 0 or epoch == 1:
        torch.save(model.state_dict(), path + "/epoch" + str(epoch) + ".pth")