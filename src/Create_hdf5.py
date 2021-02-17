import numpy as np
import os
import multiprocessing
import h5py

from tqdm import tqdm
from glob import glob
from path import Path
from functools import partial
import Preprocess_image as pp

class Sample():
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label

class Dataset():
    def __init__(self, raw_path):

        assert Path(raw_path).exists()

        self.dataset = {
                'train': {
                    'image':[],
                    'label':[], 
                    'augmentation':True
                    }, 
                'test': {
                    'image':[], 
                    'label':[], 
                    'augmentation':False
                    },
                'valid': {
                    'image':[], 
                    'label':[], 
                    'augmentation':False
                    }
                }
        self.samples = []
        self.imgdir = os.path.join(raw_path, "words")
        self.label_path = os.path.join(raw_path, "words.txt")
        self.partitions = ['train', 'test', 'valid']

    def make_partitions(self):
        label = open(self.label_path).read().splitlines()
        for line in label:
            if line[0] == '#' or not line:
                continue
            lineSplit = line.strip().split()
            if len(lineSplit) < 9:
                continue
            transcription = pp.preprocess_label(" ".join(lineSplit[8:]), self.maxTextLength)
            if transcription[0]:
                transcription = transcription[1]
            else:
                continue
            fileNameSplit = lineSplit[0].split('-')
            fileName = os.path.join(self.imgdir, fileNameSplit[0], "-".join(fileNameSplit[0:2]), lineSplit[0]) + ".png"
            sample = Sample(fileName, transcription)
            self.samples.append(sample)

        np.random.shuffle(self.samples)

        splitIdx1 = {'train':0, 'test':0, 'valid':0}
        splitIdx2 = {'train':0, 'test':0, 'valid':0}
        splitIdx1['test'] = splitIdx2['train'] = int(0.8 * len(self.samples))
        splitIdx2['test'] = splitIdx1['valid'] = int(0.9 * len(self.samples))
        splitIdx2['valid'] = len(self.samples)

        dataset = self.dataset
        for p in self.partitions:
            dataset[p]['image'] += [sample.file_path for sample in self.samples[splitIdx1[p]:splitIdx2[p]]]
            dataset[p]['label'] += [sample.label for sample in self.samples[splitIdx1[p]:splitIdx2[p]]]

        return dataset

    def read_partitions(self):
        dataset = self.make_partitions()

        for p in self.partitions:
            self.dataset[p]['image'] += dataset[p]['image']
            self.dataset[p]['label'] += dataset[p]['label']


    def save_partitions(self, target, target_image_shape, maxTextLength=32):
        self.maxTextLength = maxTextLength
        self.read_partitions()

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, 'w') as hf:
            for p in self.partitions:
                size = (len(self.dataset[p]['image']), ) + target_image_shape[:2]
                total += size[0]

                hf.create_dataset(f"{p}/image", size, dtype=np.uint8, compression='gzip', compression_opts=9)
                hf.create_dataset(f"{p}/label", (size[0],), dtype=f"S{maxTextLength}", compression='gzip', compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for p in self.partitions:
            for batch in range(0, len(self.dataset[p]['image']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(pp.preprocess_image, target_size=target_image_shape, augmentation=self.dataset[p]['augmentation']), self.dataset[p]['image'][batch:batch+batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{p}/image"][batch:batch+batch_size] = images
                    hf[f"{p}/label"][batch:batch+batch_size] = [s.encode() for s in self.dataset[p]['label'][batch:batch+batch_size]]
                    pbar.update(batch_size)

