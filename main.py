from train import SRTrain

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datamodule import SRDataModule

import yaml
from utils import *

def main(args):
    pl.seed_everything(config['random_seed'], workers=True)
    se_datamodule = SRDataModule(config)
    se_train = SRTrain(config)
    
    check_dir_exist(config['train']['output_dir_path'])
    check_dir_exist(config['train']['logger_path'])

    tb_logger = pl_loggers.TensorBoardLogger(config['train']['logger_path'], name=f"SR_logs")


    tb_logger.log_hyperparams(config)
    
    checkpoint_callback = ModelCheckpoint(
    filename = "{epoch}-{val_loss:.4f}",
    save_top_k = 1,
    mode = 'min',
    monitor = "val_loss"
    )
    
    early_stopping = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 6,
        verbose = True
    )
    
    
    
    trainer=pl.Trainer(devices=config['train']['devices'], accelerator="gpu", strategy='ddp',
    max_epochs=config['train']['total_epoch'],
    callbacks= [checkpoint_callback, early_stopping],
    logger=tb_logger,
    profiler = "simple"
    )

    trainer.fit(se_train, se_datamodule)

if __name__ == "__main__":

    config = yaml.load(open("./config.yaml", 'r'), Loader=yaml.FullLoader)
    main(config)