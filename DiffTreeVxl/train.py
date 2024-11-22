import argparse
import yaml
import os

from model.model_trainer import DiffusionModel
from model.data_loader import TreeDataLoader
# from rebuttal.data_loader import TreeDataLoader

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
# from pytorch_lightning.strategies import DDPStrategy

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    config = yaml.safe_load(string)
    return config

def main(opt):
    config = read_yaml(opt.config)
    # print(config['batch'])
    os.environ["WANDB_API_KEY"] = config['wandb']['wandb_key']
    seed_everything(config['seed'])
    model = DiffusionModel(
        base_channels=config['network']['base_channels'],
        lr=config['train']['lr'],
        batch_size=config['train']['batch_size'],
        optimizier=config['train']['optimizier'],
        scheduler=config['train']['scheduler'],
        ema_rate=config['train']['ema_rate'],
        verbose=config['verbose'],
        img_backbone=config['network']['img_backbone'],
        dim_mults=config['network']['dim_mults'],
        training_epoch=config['train']['training_epoch'],
        gradient_clip_val=config['train']['gradient_clip_val'],
        noise_schedule=config['train']['noise_schedule'],
        img_size=config['data']['img_size'],
        image_condition_dim=config['network']['img_backbone_dim'],
        dropout=config['network']['dropout'],
        with_attention=config['network']['with_attention']
    )

    data_loader = TreeDataLoader(
        data_dir=config['data']['train_path'],
        batch_size=config['train']['batch_size'],
        img_size=config['data']['img_size'],
        debug=config['train']['debug'],
    )
    train_data = data_loader.train_dataloader()
    val_data = data_loader.val_dataloader()

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=config['results']['results_folder'],
        filename="img-{epoch:02d}-{loss:.4f}",
        save_top_k=config['results']['save_top_k'],
        save_last=config['results']['save_last'],
        mode="min",
    )

    wandb_logger = WandbLogger(project="LatenTree")
    # ckpt_path = "rebuttal/results/last.ckpt"
    trainer = Trainer(devices=config['train']['devices'],
                      accelerator="gpu",
                      strategy="ddp",
                      logger=wandb_logger,
                      max_epochs=config['train']['training_epoch'],
                      log_every_n_steps=10,
                      callbacks=[checkpoint_callback],
                      # limit_train_batches=0.5,
                      )

    trainer.fit(model,
                train_data,
                val_data,
                # ckpt_path=ckpt_path,
                )


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    parse = argparse.ArgumentParser()
    # parse.add_argument('--config', type=str, default='./rebuttal/depth.yaml')
    parse.add_argument('--config', type=str, default='./rebuttal/depth.yaml')
    opt = parse.parse_args()
    main(opt)