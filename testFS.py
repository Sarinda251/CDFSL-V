from VideoDataset import FrameSampleDataset
from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
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

outfile = "test_"
    
logging.basicConfig(filename=outfile + ".log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug('logged in')

def FewShotEp(model, dataset, num_classes):

    fewClasses = []
    for i in range(5):
        n = random.randint(1,num_classes)
        fewClasses.append(n)
    
    train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=5)
    
    supportX = []
    supportY = []
    queryX = []
    queryY = []
    
    clf = sklearn.linear_model.LogisticRegression(random_state=0,
                                                  solver='newton-cg',
                                                  max_iter=1000,
                                                  C=1,
                                                  multi_class='multinomial')
    
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
        for idx in idxClass[:kshot]:
            #vid = dataset.__getitem__(idx)
            supportX.append(dataset.__getitem__(idx)['video'])
            supportY.append(i)
        for idx in idxClass[kshot:]:
            queryX.append(idx)
            queryY.append(i)
            
        #print(supportY)
        #print(queryY)
    
    supportX = torch.stack(supportX)#.to(device)
    supportXout = []
    for i in range(0, len(supportX), 5):
        supportXiter = supportX[i:i+5].to(device)
        supportXout.append(model(supportXiter).logits)
    
    support = torch.cat(supportXout)
    supportY = torch.tensor(supportY)
    #print(supportX.shape)
    
    #support = model(supportX).logits
    del supportX
    support_features_np = support.data.cpu().numpy()                                                  
    support_ys_np = supportY.data.cpu().numpy()
    clf.fit(support_features_np, support_ys_np)
    
    del support_features_np
    
    valacc = []
    
    perm = np.random.permutation(len(queryY))
    #print(len(queryY))
    queryY = [queryY[i] for i in perm]
    queryX = [queryX[i] for i in perm]
    nAtAtime = 5
    for i in range(0, len(queryY), nAtAtime):
        #print(queryY[i:i+nAtAtime])
        if len(queryY[i:i+nAtAtime]) > 0:
            #vid = dataset.__getitem__(idx)
            #queryvids = [dataset.__getitem__(queryid)['video'] for queryid in queryX[i:i+nAtAtime]]
            queryX2 = torch.stack([dataset.__getitem__(queryid)['video'] for queryid in queryX[i:i+nAtAtime]]).to(device)
            queryY2 = torch.tensor(queryY[i:i+nAtAtime])
            outputs = model(queryX2).logits
            outputs = clf.predict(outputs.data.cpu().numpy())
            acc = train_acc(torch.from_numpy(outputs), queryY2.cpu())
            valacc.append(acc.item())
            del queryX2
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

outfile = "test_" + args.target

logging.basicConfig(filename=outfile + ".log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug('logged in')

with open('dataset/datapath.json') as json_file:
    datapath = json.load(json_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torchvision.set_video_backend('video_reader')
random.seed(251)
warnings.simplefilter('ignore')

def Average(lst):
    return sum(lst) / len(lst)

#frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)
#frame_transform = torchvision.transforms.Compose(transformsf)
#video_transform = torchvision.transforms.Compose(transformsv)
#source_set = FrameSampleDataset("dataset/ucf101_train_clean.txt", frame_transform=frame_transform)#, video_transform = video_transform)
#targetloader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, num_workers=args.workers, shuffle = True, pin_memory = True, drop_last=True)

if args.model == "MAEs":
    frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)

    configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
    configuration.num_hidden_layers = 8
    configuration.intermediate_size = 2358
    configuration.image_size = 112
    configuration.num_labels = 61
    logger.debug("config loaded")

    model = VideoMAEForVideoClassification(configuration)
elif args.model == "MAEsU":
    frame_transform = VideoMAEImageProcessor(size= {"height": 112, "width": 112}, do_center_crop=False)

    configuration = VideoMAEConfig()#'num_hidden_layers' = 8, 'intermediate_size' = 2358)
    configuration.num_hidden_layers = 8
    configuration.intermediate_size = 2358
    configuration.image_size = 112
    configuration.num_labels = 101
    logger.debug("config loaded")

    model = VideoMAEForVideoClassification(configuration)
elif args.model == "MAEdefault":
    configuration = VideoMAEConfig()
    model = VideoMAEForVideoClassification(configuration)
    frame_transform = VideoMAEImageProcessor(size= {"height": 224, "width": 224}, do_center_crop=False)
else:
    print('invalid model architecture')
    exit(0)

logger.debug("model configured")


target_set = FrameSampleDataset(datapath[args.target]['pathTest'], frame_transform=frame_transform, clip_len = args.num_frames, logger = logger)
logger.debug("data loaded")


if args.load:
    model.load_state_dict(torch.load(args.load), strict=False)
    logger.debug('loaded from: ' + args.load)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
        
model.classifier = Identity()
model.eval()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print("using " + str(torch.cuda.device_count()) + " gpus")

model.to(device)

accs = []
for i in range(20):                                
    accs.append(FewShotEp(model=model, dataset=target_set, num_classes=datapath[args.target]['num_classes']))
    
print(Average(accs))

'''if __name__ == '__main__':
    parser.add_argument('--batch_size', default=5)
    parser.add_argument('--model', default="MAEs")
    args = parser.parse_args()
    
    evaluate(args=args)'''
        