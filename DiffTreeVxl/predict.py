from PIL import Image
from model.model_trainer import DiffusionModel
from model.utils import *
from torchvision import transforms

import open3d as o3d

# data_folder = "C:/Users/13247/Desktop/predict/dataset/1115.png"
data_folder = "D:/OnePiece/graduate/Reconstruction/230626-LatentTree/natural_img/65.png"

ckpt = "./results/vit-cos-atten-epoch=2131-loss=1.3420.ckpt"
config = read_yaml('config/vit-cos-atten-config.yaml')

# ckpt = "./results/r50-cosine-epoch=911-loss=0.1422.ckpt"
# config = read_yaml('config/r50-cos-config.yaml')


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
    image_condition_dim=config['network']['img_backbone_dim'],
    dropout=config['network']['dropout'],
    with_attention=config['network']['with_attention']
).load_from_checkpoint(ckpt).cuda()

img = Image.open(data_folder).convert('RGB')
# img.show()
img_size = config['data']['img_size']

transform = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img = transform(img).unsqueeze(0)
print(img.shape)

voxel = model.sample_with_img(img.cuda(), steps=21, verbose=True)
# print(voxel.shape)
size = 64
voxel = voxel.reshape([size, size, size])
voxel = voxel.cpu().numpy()
np.save("C:/Users/13247/Desktop/last.npy", voxel)

xyz = []
rgb = []
for x in range(size):
    for y in range(size):
        for z in range(size):
            if voxel[x,y,z] > 0.:
                xyz.append([x,y,z])
                if voxel[x,y,z] > 0.8:
                    rgb.append([1, 0, 0])
                elif voxel[x,y,z] > 0.3:
                    rgb.append([0.3, 0, 0])
                elif voxel[x,y,z] > 0:
                    rgb.append([0, 0.5, 0])

PC = o3d.geometry.PointCloud()
PC.points = o3d.utility.Vector3dVector(xyz)
PC.colors = o3d.utility.Vector3dVector(rgb)

# model = o3d.geometry.PoiPC, PCpntCloud(PC)
model = o3d.geometry.VoxelGrid.create_from_point_cloud(PC, voxel_size=1)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0)
o3d.visualization.draw_geometries([model, coord_frame])