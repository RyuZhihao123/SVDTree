import os
import copy

import timm.models.swin_transformer_v2
import torch
import torch.nn as nn
import torchvision.models as models

from torch.optim import AdamW, Adam
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from timm import create_model
import os
from model.utils import *


class UNetModel(nn.Module):
    def __init__(self,
                 img_backbone: str = 'Resnet50',
                 base_channels: int = 16,
                 dim_mults=(1, 2, 4, 8, 16),
                 dropout: float = 0.1,
                 img_size: int = 224,
                 image_condition_dim: int = 512,
                 with_attention: bool = False,
                 verbose: bool = False,
                 ):
        super().__init__()
        self.verbose = verbose
        if img_backbone == 'Resnet50':
            # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(2048, image_condition_dim).apply(weights_init_normal)
            # # todo: rebuttal
            # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif img_backbone == 'Swin2':
            self.backbone = create_model("swinv2_base_window8_256", pretrained=True,
                                         pretrained_cfg_overlay=dict(file='./checkpoint'))
            # self.backbone = models.swin_v2_b(pretrained=True)
            self.backbone.head = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        elif img_backbone == 'Vit':
            # 之前的一版vit用的是models.vit_b_16 768维， 目前加入attention的一版vit用的是models.vit_l_16 1024维
            # self.backbone = create_model('vit_base_patch16_224', pretrained=True)
            # self.backbone.heads = nn.Linear(768, image_condition_dim).apply(weights_init_normal)
            self.backbone = models.vit_l_16(pretrained=True)
            self.backbone.heads = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        else:
            raise NotImplementedError

        self.img_size = img_size
        channels = [base_channels, *map(lambda m: base_channels * m, dim_mults)]
        if self.verbose:
            print(channels)
        emb_dim = base_channels * 4
        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        ).apply(weights_init_normal)

        self.input_emb = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        down_num = len(channels)-1
        size = 64
        for idx in range(down_num):
            if with_attention:
                # print("****************" + str(size))
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_list=[channels[idx],
                                          channels[idx] // 2,
                                          channels[idx] // 2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout),
                    MixImgAttention(channels[idx],
                                    channels[idx + 1],
                                    img_dim=image_condition_dim,
                                    voxel_size=size,
                                    dropout=dropout)
                ]))
                size = size // 2
            else:
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_list=[channels[idx],
                                          channels[idx]//2,
                                          channels[idx]//2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout),
                    MixImgFeature(channels[idx],
                                  channels[idx+1],
                                  img_dim=image_condition_dim,
                                  dropout=dropout)
                ]))

        self.mid_block = ResnetBlock(
            dim_list=[channels[-1], channels[-1] // 2, channels[-1] // 2, channels[-1]],
            time_dim=emb_dim,
            dropout=dropout)
        channels = channels[::-1]
        if self.verbose:
            print(channels)
        for idx in range(down_num):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[idx], channels[idx + 1],
                                       kernel_size=4, stride=2, padding=1),
                    normalization(channels[idx + 1]),
                    activation_function(),
                ).apply(weights_init_normal)
            )
        self.out = nn.Sequential(
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # todo: rebuttal
        # self.rebuttal = nn.Conv2d(5, 3, 1, 1).apply(weights_init_normal)

    def forward(self, x, t, img):
        # todo: rebuttal
        # img = self.rebuttal(img)

        img = self.backbone(img)
        x = self.input_emb(x.to(torch.float32))
        t = self.time_emb(self.time_pos_emb(t))
        if self.verbose:
            print(x.shape, t.shape)
        h = []
        for resnet, mix in self.downs:
            x = resnet(x, t)
            x = mix(x, img)
            if self.verbose:
                print(x.shape)
            h.append(x)
        x = self.mid_block(x, t)
        if self.verbose:
            print(x.shape)
        for upblock in self.ups:
            x = upblock(x + h.pop())
            if self.verbose:
                print(x.shape)
        x = self.out(x)
        if self.verbose:
            print(x.shape)
        return x

# m = UNetModel(with_attention=True, verbose=True)
# a = torch.randn((5, 1, 64, 64, 64))
# b = torch.randn((5))
# c = torch.randn((5, 5, 224, 224))
# m(a, b, c)


class DiffusionModel(LightningModule):
    def __init__(
        self,
        base_channels: int = 64,
        lr: float = 2e-4,
        batch_size: int = 8,
        optimizier: str = "adam",
        scheduler: str = "CosineAnnealingLR",
        ema_rate: float = 0.999,
        verbose: bool = False,
        img_backbone: str = 'Resnet50',
        dim_mults=(1, 2, 4, 8, 16),
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        noise_schedule: str = "linear",
        img_size: int = 8,
        image_condition_dim: int = 512,
        dropout: float = 0.1,
        with_attention: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = UNetModel(
            img_backbone=img_backbone,
            base_channels=base_channels,
            dim_mults=dim_mults,
            dropout=dropout,
            img_size=img_size,
            image_condition_dim=image_condition_dim,
            with_attention=with_attention,
            verbose=verbose)

        self.batch_size = batch_size
        self.lr = lr
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)

        self.image_feature_drop_out = dropout
        self.optim = optimizier
        self.scheduler = scheduler
        self.eps = eps
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

    def training_loss(self, img, img_features):
        batch = img.shape[0]

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times).to(torch.float32)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        pred = self.model(noised_img, noise_level, img_features)
        img = img.to(torch.float32)
        return 100 * F.mse_loss(pred, img)

    def training_step(self, batch, batch_idx):
        voxel = batch["voxel"].unsqueeze(1)
        img_features = batch["img"]
        loss = self.training_loss(voxel, img_features).mean()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        opt.step()
        self.update_EMA()
        self.log("train_loss", loss.clone().detach().item(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        voxel = batch["voxel"].unsqueeze(1).to(torch.float32)
        img_features = batch["img"]
        # print(img_features.shape, voxel.shape)
        # pred = self.sample_with_img(
        #     img_features,
        #     batch_size=self.batch_size,
        #     steps=10)
        # # print(pred.shape)
        # loss = 100 * F.mse_loss(pred, voxel)
        loss = self.training_loss(voxel, img_features).mean()
        self.log("loss", loss.clone().detach().item(), prog_bar=True)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optim == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        if self.scheduler == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif self.scheduler == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        else:
            raise NotImplementedError
        return [optimizer], [lr_scheduler]

    @staticmethod
    def get_sampling_timesteps(batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def sample_with_img(self,
                        img,
                        batch_size=1,
                        steps=50,
                        truncated_index: float = 0.0,
                        img_weight: float = 1.0,
                        verbose: bool = False):
        vxl_size = 64
        shape = (batch_size, 1, vxl_size, vxl_size, vxl_size)
        time_pairs = self.get_sampling_timesteps(
            batch=batch_size, device=self.device, steps=steps)
        voxel = torch.randn(shape, device=self.device)
        x_start = None
        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs

        for time, time_next in _iter:
            log_snr = self.log_snr(time).type_as(time)
            log_snr_next = self.log_snr(time_next).type_as(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, voxel), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time).type_as(time)

            x_zero_none = self.model(voxel, noise_cond, img)
            x_start = x_zero_none + img_weight * \
                      (self.model(voxel, noise_cond, img) - x_zero_none)

            c = -torch.expm1(log_snr - log_snr_next)
            mean = alpha_next * (voxel * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(voxel),
                torch.zeros_like(voxel)
            )
            voxel = mean + torch.sqrt(variance) * noise
        return voxel

