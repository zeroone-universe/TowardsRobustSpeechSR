from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import os

import pytorch_lightning as pl

from utils import *

class SRDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_lr, path_dir_hr, frame_size, hop_size):
        self.path_dir_lr   = path_dir_lr
        self.path_dir_hr   = path_dir_hr  

        self.wavs = []

        self.upsample = ta.transforms.Resample(8000, 16000)
        
        paths_wav_lr= get_wav_paths(self.path_dir_lr)
        paths_wav_hr = get_wav_paths(self.path_dir_hr)
        
        for wav_idx, (path_wav_hr, path_wav_lr) in enumerate(zip(paths_wav_hr, paths_wav_lr)):
            print(f'\r{wav_idx} th file loaded', end='')
            wav_lr, _ = ta.load(path_wav_lr)
            wav_hr, _ = ta.load(path_wav_hr)
            
            wav_lr = self.upsample(wav_lr)
            if wav_hr.shape[-1] % 2 == 1:
                wav_hr = wav_hr[:, :-1]
            
            wav_lr_seg = segmentation(wav_lr, frame_size, hop_size)
            wav_hr_seg = segmentation(wav_hr, frame_size, hop_size)
            
            for idx in range(wav_hr_seg.shape[0]):
                self.wavs.append([wav_lr_seg[idx], wav_hr_seg[idx]])
    
    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.wavs)
    
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        return self.wavs[idx]
        

class SRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.data_dir = config['dataset']['data_dir']
        
        self.path_dir_lr_train = config['dataset']['lr_train']
        self.path_dir_lr_val = config['dataset']['lr_val']
        
        self.path_dir_hr_train =  config['dataset']['hr_train']
        self.path_dir_hr_val =  config['dataset']['hr_val']

        self.frame_size = config["dataset"]["frame_size"]
        self.hop_size = config["dataset"]["hop_size"]
        
        self.batch_size = config['dataset']['batch_size']
        self.num_workers = config['dataset']['num_workers']

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.train_dataset = SRDataset(
            path_dir_lr = os.path.join(self.data_dir, self.path_dir_lr_train),
            path_dir_hr = os.path.join(self.data_dir, self.path_dir_hr_train),
            frame_size = self.frame_size,
            hop_size = self.hop_size
            )


        self.val_dataset = SRDataset(
            path_dir_lr = os.path.join(self.data_dir, self.path_dir_lr_val),
            path_dir_hr = os.path.join(self.data_dir, self.path_dir_hr_val),
            frame_size = self.frame_size,
            hop_size = self.hop_size
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        pass