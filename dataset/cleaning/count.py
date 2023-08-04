import os
import csv
import io
import random
import itertools
from operator import itemgetter


#csvFile = open('../D48_test.txt', 'r')
csvFile = open('RA_over50.txt', 'r')
#csvFile2 = open('RA_over50.txt', 'w')
out = open('RA_train2.txt', 'w')
out2 = open('RA_test2.txt', 'w')
csvFile2 = open('RA_over50_2.txt', 'w')

counts = set()

for line in csvFile:
    classnum = line.strip().split(", ")[1]
    counts.add(classnum)


#for line in csvFile2:
#    classnum = line.strip().split(", ")[1]
#    counts[int(classnum)] += 1

print(len(counts))
counts = sorted(list(counts))
csvFile = open('RA_over50.txt', 'r')
for line in csvFile:
    classnum = line.strip().split(", ")[1]
    classnum = counts.index(classnum)
    csvFile2.write(line.strip().split(", ")[0] + ", " + str(classnum) + '\n')
csvFile2.close()
counts = set()
csvFile = open('RA_over50_2.txt', 'r')
for line in csvFile:
    classnum = line.strip().split(", ")[1]
    counts.add(classnum)

classdict = {}

for classid in counts:
    classdict[int(classid)] = []

csvFile = open('RA_over50_2.txt', 'r')
for line in csvFile:
    classnum = line.strip().split(", ")[1]
    classdict[int(classnum)].append(line)
#print(len(classdict.keys()))
for key in sorted(classdict.keys()):
    print(len(classdict[key]))    
    random.shuffle(classdict[key])
    
    for i in range(len(classdict[key])):
        print(i)
        if i <= len(classdict[key])/2:
            out.write(classdict[key][i])
        else:
            out2.write(classdict[key][i])
    

    