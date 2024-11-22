import open3d as o3d
import numpy as np

# data = np.load("./results.npy")
data = np.load("D:/OnePiece/graduate/Reconstruction/230626-LatentTree/DiffTreeVxl/rebuttal/depth/train/vxl/data_102.npy")

resolution = 64

xyz = []
rgb = []
for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            if data[x,y,z] >= 0:
                xyz.append([x,y,z])
                if data[x,y,z] > 0.8:
                    rgb.append([1, 0, 0])
                elif data[x,y,z] > 0.3:
                    rgb.append([0.3, 0, 0])
                elif data[x,y,z] >= 0:
                    rgb.append([0, 0.5, 0])

PC = o3d.geometry.PointCloud()
PC.points = o3d.utility.Vector3dVector(xyz)
PC.colors = o3d.utility.Vector3dVector(rgb)

# model = o3d.geometry.PoiPC, PCpntCloud(PC)
model = o3d.geometry.VoxelGrid.create_from_point_cloud(PC, voxel_size=1)

o3d.visualization.draw_geometries([model])

# # 将光源添加到渲染窗口
# vis.add_geometry(model)
# vis.add_geometry(scene)
#
# # 渲染并显示
# vis.poll_events()
# vis.update_renderer()
# vis.run()


# o3d.io.write_voxel_grid("C:/Users/13247/Desktop/result.ply",
#                         voxel_grid=model,
#                         print_progress=True,)

# import torch
#
# a = torch.zeros([2, 3, 3])
# b = torch.ones([1, 3, 3])
# print(a+b)