# SVDTree.

This repo includes the implementation of our paper **SVDTree (Accepted by CVPR 2024)**.

- I am one of the joint first authors ([Zhihao Liu](https://ryuzhihao123.github.io/)), who take charge of the implementation of the **second main step** of this paper, i.e. **``Voxel (Point Cloud)-based 3D Tree Reconstruction``**.
This part is entirely implemented by myself so I released it seperately in my own Github.

- The algorithm is designed with reference to the principles of two SIGGRAPH papers [[link](https://dl.acm.org/doi/10.1145/1866158.1866177)][[link](https://dl.acm.org/doi/10.1145/1289603.1289610)] **(They are not open-source so I guess my code is the first re-implementation in the public and outperforms the original paper a lot)**.

![image info](https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_0.png)

## Point Cloud (Voxel)-based 3D Tree Reconstruction. 

Since point clouds are way more complex and common than the voxel representation used in our paper, we released a **more Generic Version** of our codes so that you can easily handle both two data representations more robustly. 


:heart: **Source Codes**: Please check this [[folder](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_Codes)] for my full source codes. 

You can use Qt 5.8+ or Visual Studio 2019+ to easily compile it. The configuration of **CMake** and **OpenGL** is also required.

Please also download this example 3D point cloud file [[xyz file](https://github.com/RyuZhihao123/SVDTree/blob/main/Tree1_input.xyz)] to quickly have a try with our software.

:heart: **My software** (User Interface):

<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_1.png" width = "700" alt="ack" title="dasdasdsa title" align=center />
<br/><center><b>Fig. 1. Input 3D point cloud.</b></center>
</div>
<br/>
<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_2.png" width = "700" alt="ack" title="dasdasdsa title" align=center />
<br/><center><b>Fig. 2. Clustering.</b></center>
</div>
<br/>
<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_3.png" width = "700" alt="ack" title="dasdasdsa title" align=center />
<br/><center><b>Fig. 3. The extracted 3D tree skeleton.</b></center>
</div>
<br/>











