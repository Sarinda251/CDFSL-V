import os
import torch
import torchvision
import itertools
import random

class FrameSampleDataset(torch.utils.data.Dataset):
    def __init__(self, annotation, epoch_size=None, frame_transform=None, video_transform=None, clip_len=8, logger=None):
        super(FrameSampleDataset).__init__()
        random.seed(251)
        #self.samples = get_samples(root)
        lines = []
        with open(annotation) as f:
            for line in f:
                split = line.strip().split(', ')
                if split[1] == 'label':
                    continue
                lines.append((split[0], str(int(split[1]) - 1)))
        
        print('start', lines[0])
        self.samples = lines
        self.logger = logger

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __getitem__(self, idx: int):
        #path, target = random.choice(self.samples)
        path, target = self.samples[idx]
        # Get video object
        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()
        video_frames = []  # video frame buffer
        
        if not metadata["video"]['duration']:
            logger.error('fail', path)
            #return  {'fail': path}
        # Seek and return frames
        max_seek = metadata["video"]['duration'][0] - ((self.clip_len + 2) / metadata["video"]['fps'][0])
        if max_seek < 0:
            dummyvar = 0
            #logger.error('fail ' + path + " " + " max seek: " + str(max_seek))
            #max_seek = 0
        start = max(random.uniform(0., max_seek), 0)
        for frame in itertools.islice(vid.seek(start), self.clip_len):
            if self.frame_transform is None:
                video_frames.append(frame['data'])
            else:
                video_frames.append(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor))
                #video_frames.append(self.frame_transform(frame['data'].type(torch.FloatTensor)))
                #print(self.frame_transform.preprocess(frame['data'], return_tensors="pt").pixel_values.squeeze(0).squeeze(0).type(torch.FloatTensor).size())
            current_pts = frame['pts']
                    
        while len(video_frames) < self.clip_len:
            video_frames.append(video_frames[-1])
            #logger.error('fail ' + path + " " + str(video.shape[0]) + " frames, start time: " + str(start) + " max seek: " + str(max_seek))
        # Stack it into a tensor
        video = torch.stack(video_frames, 0)
        #video = video.type(torch.FloatTensor)
        if self.video_transform:
            video = self.video_transform(video)

        output = {
            'path': path,
            'video': video,
            'target': target,
            'start': start,
            'end': current_pts}
        return output
    def __len__(self):
        return self.epoch_size