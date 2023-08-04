from VideoDataset import FrameSampleDataset
import random
import json
from opts import parser
import logging
import sklearn.linear_model
import torch
import torchvision
import warnings
import torch.nn as nn
import torchmetrics
import numpy as np
import logging

def FewShotEp(dataset, num_classes):

    fewClasses = []
    for i in range(5):
        n = random.randint(1,num_classes)
        fewClasses.append(n)
    
    train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=5)
    
    supportX = []
    supportY = []
    queryX = []
    queryY = []
    
    print(fewClasses)
    for i in range(len(fewClasses)):
        fewClass = fewClasses[i]
        #print(fewClass)
        #numberedlist = enumerate(dataset.samples)
        idxClass = [i for i, data in enumerate(dataset.samples) if int(data[1]) == fewClass]
        #idxClass = [i int(data[i]) = fewClass]
        #for data in dataset.samples:
        #    print(data[1], fewClass)
        #print(idxClass)
        kshot = 5
        for idx in idxClass[kshot:]:
            queryX.append(idx)
            queryY.append(i)
            
        #print(supportY)
        #print(queryY)
    
    #print(supportX.shape)
    
    #support = model(supportX).logits
    
    valacc = []
    
    perm = np.random.permutation(len(queryY))
    #print(len(queryY))
    queryY = [queryY[i] for i in perm]
    nAtAtime = 5
    for i in range(0, len(queryY), nAtAtime):
        #print(queryY[i:i+nAtAtime])
        if len(queryY[i:i+nAtAtime]) > 0:
            #vid = dataset.__getitem__(idx)
            #queryvids = [dataset.__getitem__(queryid)['video'] for queryid in queryX[i:i+nAtAtime]]
            queryY2 = torch.tensor(queryY[i:i+nAtAtime])
            print(queryY2)
            outputs = np.random.choice(5, len(queryY2))
            print(outputs)
            acc = train_acc(torch.from_numpy(outputs), queryY2.cpu())
            valacc.append(acc.item())
            del outputs
    #print(outputs)
    #acc = train_acc(outputs.argmax(dim=-1).cpu(), labels.cpu())
    print(Average(valacc))
    return Average(valacc)




parser.add_argument('--batch_size', default=5)
parser.add_argument('--model', default="MAEs")
args = parser.parse_args()
#parser.add_argument('--load', default='ucf')
#parser.add_argument('--target', default='hmdb51')

if not args.load:
    args.load = ""

with open('dataset/datapath.json') as json_file:
    datapath = json.load(json_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

random.seed(251)
warnings.simplefilter('ignore')

def Average(lst):
    return sum(lst) / len(lst)

#frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
#frame_transform = torchvision.transforms.Compose(transformsf)
#video_transform = torchvision.transforms.Compose(transformsv)
#source_set = FrameSampleDataset("dataset/ucf101_train_clean.txt", frame_transform=frame_transform)#, video_transform = video_transform)
#targetloader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=args.workers, shuffle = True, pin_memory = True, drop_last=True)


target_set = FrameSampleDataset(datapath[args.target]['pathTest'], frame_transform=None, clip_len = args.num_frames)

accs = []
for i in range(20):                                
    accs.append(FewShotEp(dataset=target_set, num_classes=datapath[args.target]['num_classes']))
    
print(Average(accs))

'''if __name__ == '__main__':
    parser.add_argument('--batch_size', default=5)
    parser.add_argument('--model', default="MAEs")
    args = parser.parse_args()
    
    evaluate(args=args)'''
        