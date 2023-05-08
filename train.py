import pytorch_lightning as pl
import torch
import sys

import os
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio as ta
from loss import *
from utils import *
from AECNN import AECNN

class SRTrain(pl.LightningModule):
    def __init__(self, config):
        super(SRTrain, self).__init__()
        self.automatic_optimization = False
        self.config = config
        
        self.resampler = T.Resample(8000, 16000, dtype = torch.float32)
        self.aecnn = AECNN(config)
        
        self.criterion =  getattr(sys.modules[__name__], config["loss"]["type"])(config)
        
        #optimizer & scheduler parameters
        self.initial_lr = config['optim']['initial_lr']
        self.patience_epoch = config['optim']['patience_epoch']   
             
        self.frame_size = config["dataset"]["frame_size"]
        self.hop_size = config["dataset"]["hop_size"]
        
        #Sample for logging
        self.data_dir = config['dataset']['data_dir']
        self.path_dir_lr_val = config['dataset']['lr_val']
        self.path_dir_hr_val =  config['dataset']['hr_val']
        
        self.output_dir_path = config['train']['output_dir_path']
        
        self.path_sample_lr, self.path_sample_hr = get_one_sample_path(dir_lr_path= os.path.join(self.data_dir, self.path_dir_lr_val), dir_hr_path=os.path.join(self.data_dir, self.path_dir_hr_val))
        
    def forward(self,x):
        output = self.aecnn(x)
        return output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.aecnn.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor=0.5, patience=self.patience_epoch, verbose=True)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        wav_lr, wav_hr = batch
        wav_sr = self.forward(wav_lr)
        
        loss = self.criterion(wav_lr, wav_sr, wav_hr)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        self.log("train_loss", loss,  prog_bar = True, batch_size = self.config['dataset']['batch_size'])


    def validation_step(self, batch, batch_idx):
        wav_lr, wav_hr = batch
        wav_sr = self.forward(wav_lr)
        
        loss = self.criterion(wav_lr, wav_sr, wav_hr)

        self.log("val_loss", loss, batch_size = self.config['dataset']['batch_size'], sync_dist=True)
        
    def on_validation_epoch_end(self):
        
        sample_lr, _  = ta.load(self.path_sample_lr)
        sample_hr, _ = ta.load(self.path_sample_hr)
        sample_lr = sample_lr.to(self.device)
        sample_hr =sample_hr.to(self.device)
        
        sample_sr = self.synth_one_sample(sample_lr)
        sample_sr = sample_sr.cpu()
        
        ta.save(f"{self.output_dir_path}/sample_{self.current_epoch}.wav", sample_sr, 16000)
        
        
        scheduler = self.lr_schedulers()
        scheduler.step(self.trainer.callback_metrics['val_loss'])
        
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def synth_one_sample(self, wav):
        wav = self.resampler(wav)
        
        wav = wav.unsqueeze(1)
        wav_padded = F.pad(wav, (0, self.frame_size), "constant", 0)
        wav_seg = wav_padded.unfold(-1,self.frame_size, self.hop_size)
        B, C, T, L = wav_seg.shape
        
        wav_seg = wav_seg.transpose(1,2).contiguous()
        wav_seg = wav_seg.view(B*T, C, L) 

        wav_seg = self.forward(wav_seg)
        wav_seg.view(B,T,C,L).transpose(1,2).contiguous()
        wav_seg = wav_seg.view(B, C*T, L)
        
        wav_rec = F.fold(
            wav_seg.transpose(1,2).contiguous()*torch.hann_window(self.frame_size, device = wav_seg.device).view(1, -1, 1),
            output_size = [1, (wav_seg.shape[-2]-1)*self.hop_size + self.frame_size],
            kernel_size = (1, self.frame_size),
            stride = (1, self.hop_size)
        ).squeeze(-2)
        
        wav_rec = wav_rec / (self.frame_size/(2*self.hop_size))
        wav_rec = wav_rec.squeeze(0)
        
        return wav_rec