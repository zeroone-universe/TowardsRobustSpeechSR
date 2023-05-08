import torch

#-------------------------------
#RI-MAG
#-------------------------------

class LossRIMAG:
    def __init__(self, config):
        self.loss_f = LossF(config)
        self.loss_ri = LossRI(config)
        
    def __call__(self, x_lr, x_sr, x_hr):
        return self.loss_f(x_sr, x_hr) + self.loss_ri(x_sr, x_hr)
        

#-------------------------------
#TF Loss
#-------------------------------

class LossTF:
    def __init__(self, config):
        self.loss_t = LossT(config) 
        self.loss_f = LossF(config)
        self.tf_alpha = config['loss']['tf_alpha']
        
    def __call__(self, x_lr, x_sr, x_hr):
        return self.tf_alpha*self.loss_t(x_sr, x_hr) + (1-self.tf_alpha)*self.loss_f(x_sr, x_hr)
    
#-------------------------------
#T-PCM Loss
#-------------------------------

class LossTPCM:
    def __init__(self, config):
        self.loss_t = LossT(config)
        self.loss_pcm = LossPCM(config)
        self.tpcm_beta = config['loss']['tpcm_beta']
        
    def __call__(self, x_lr, x_sr, x_hr):
        return self.tpcm_beta * self.loss_t(x_sr, x_hr) + (1-self.tpcm_beta)*self.loss_pcm(x_lr, x_sr, x_hr)
        


#-------------------------------
#Ingredient Loss
#-------------------------------

#T Loss

class LossT:
    def __init__(self, config):
        pass
    
    def __call__(self, x_sr, x_hr):
        return torch.mean(torch.abs(x_sr - x_hr))

#F Loss

class LossF:
    def __init__(self, config):
        self.stft_mag = stft_mag(
            nfft = config['loss']['window_size'],
            window_size = config['loss']['window_size'],
            hop_size = config['loss']['hop_size']
        )
    
    def __call__(self, x_sr, x_hr):
        total_num = x_sr.shape[0]
        total_loss = 0
        
        for idx in range(total_num):
            x_noisy = x_sr[idx]
            x_target = x_hr[idx]
            loss = torch.mean(torch.abs(self.stft_mag(x_target) - self.stft_mag(x_noisy)))
            total_loss += loss
        
        return total_loss/total_num
        
class stft_mag:
    def __init__(self, nfft, window_size, hop_size):
        self.nfft = nfft
        self.window_size = window_size
        self.hop_size = hop_size
        
    def __call__(self, x):
        window = torch.hann_window(self.window_size).to(x.device)
        x_stft = torch.stft(x, n_fft = self.nfft, hop_length=self.hop_size, win_length=self.window_size,
                    window = window, return_complex=True)
        return abs(x_stft)
        
      
#RI loss

class LossRI:
    def __init__(self, config):
        self.stft_risum = stft_RIsum(
            nfft = config['loss']['window_size'],
            window_size = config['loss']['window_size'],
            hop_size = config['loss']['hop_size']
        )

    def __call__(self, x_sr, x_hr):
        
        total_num = x_sr.shape[0]
        total_loss = 0
        
        for idx in range(total_num):
            x_noisy = x_sr[idx]
            x_target = x_hr[idx]
            loss = torch.mean(torch.abs(self.stft_risum(x_target) - self.stft_risum(x_noisy)))
            total_loss+=loss
   
        return total_loss/total_num

class stft_RIsum:
    def __init__(self, nfft, window_size, hop_size):
        self.nfft = nfft
        self.window_size = window_size
        self.hop_size = hop_size
        
    def __call__(self, x):
        
        window = torch.hann_window(self.window_size).to(x.device)
        x_stft = torch.stft(x, n_fft = self.nfft, hop_length=self.hop_size, win_length=self.window_size,
                    window = window, return_complex=True)
        real = x_stft[...,0]
        imag = x_stft[...,1]
        
        return torch.abs(real) + torch.abs(imag)
    

#PCM Loss
class LossPCM:
    def __init__(self, config):
        self.loss_f1 = LossF(config)
        self.loss_f2 = LossF(config)
    
    def __call__(self, x_lr, x_sr, x_hr):
        return self.loss_f1(x_sr, x_hr) + self.loss_f2(x_sr - x_lr, x_hr - x_lr)