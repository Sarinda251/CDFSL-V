import os
import csv
from torchvision.io import VideoReader
import io
import random
import itertools
import torchvision
from operator import itemgetter
torchvision.set_video_backend('video_reader')

csvFile = open('rareactcutall.csv', 'r')
outfile = open('RA_full.txt', 'w')

vidlist = []

for line in csvFile:
    line = line.strip().split(',')
    line[0] = "/home/c3-0/al209167/datasets/RareActCutAll/" + line[0]
    line[1] = int(line[1])
    
    
    vid = VideoReader(line[0], "video")
    metadata = vid.get_metadata()
    video_frames = []  # video frame buffer
      
    if not metadata["video"]['duration']:
        print('fail', line[0])
    else:
        #print(metadata["video"]['fps'], metadata["video"]['duration'])
        print(line[0])
        max_seek = metadata["video"]['duration'][0] - (16 / metadata["video"]['fps'][0])
        if max_seek >= 0:
            print("good")
            #f.write(parts[i] + ", " + labels[i] + "\n")
            max_seek = metadata["video"]['duration'][0] - (16 / metadata["video"]['fps'][0])
            if metadata["video"]['duration'][0] < 0:
                max_seek = 16
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), 16):
                video_frames.append(frame['data'])
            if len(video_frames) == 16:
                vidlist.append((line[0], line[1]))
        else:
            print('fail ' + line[0] + " " + str(max_seek + 16) + "frames")
    
print("list compiled")
vidlist = sorted(vidlist, key=itemgetter(1))
print("list sorted")
outfile.write(vidlist[0][0] + ", " + str(vidlist[0][1]))
for vidtup in vidlist:
    outfile.write("\n" + vidtup[0] + ", " + str(vidtup[1]))


