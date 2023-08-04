import os
import csv
from torchvision.io import VideoReader
import io
import random
import itertools
import torchvision
from operator import itemgetter
torchvision.set_video_backend('video_reader')

csvFile = open('../SSmini_test.txt', 'r')
outfile = open('SSmini_test_order.txt', 'w')

vidlist = []

minInd = 99999

for line in csvFile:
    line = line.strip().split(', ')
    line[0] = line[0]
    line[1] = int(line[1])
    minInd = min(minInd, line[1])
    vidlist.append((line[0], line[1]))
    print(line[0])
    
print("list compiled")
vidlist = sorted(vidlist, key=itemgetter(1))
print("list sorted")
if minInd == 0:
    outfile.write(vidlist[0][0] + ", " + str(1 + vidlist[0][1]))
    for vidtup in vidlist[1:]:
        outfile.write("\n" + vidtup[0] + ", " + str(1 + vidtup[1]))
elif minInd == 0:
    outfile.write(vidlist[0][0] + ", " + str(vidlist[0][1]))
    for vidtup in vidlist[1:]:
        outfile.write("\n" + vidtup[0] + ", " + str(vidtup[1]))
else:
    print('label error')