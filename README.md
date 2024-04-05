# SVDTree.

This repo includes the program of our paper **SVDTree (Accepted by CVPR 2024)**.

- I am one of the joint first authors ([Zhihao Liu](https://ryuzhihao123.github.io/)), who take charge of the implementation of the **second main step** of this paper, i.e. **``Voxel (Point Cloud)-based 3D Tree Reconstruction``**.
This part is entirely implemented by myself so I released it seperately in my own Github.


![image info](https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_0.png)


## Point Cloud (Voxel)-based 3D Tree Reconstruction. 

Since point clouds are way more complex and common than the voxel representation used in our paper, we released a **more Generic Version** of our codes so that you can easily handle both two data representations more robustly. 

- 🔵 **Executable Program:** Please download the entire folder under this **[[LINK](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe)]**. If you're using **Win10+**, we also strongly suggest you to try **the EXE version** that we released for quick start (推荐首先尝试该exe程序：无需任何编译+可直接双击使用). After downloading the entire [directory](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe), you can directly run the program by double clicking the **``TreeFromPoint.exe``**.

- 🟢 **Codes**: Please check this [**[Folder](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes)**] to download my source codes. 

- 🟠 **Test Data of a 3D Point Cloud:** Please also download **this EXAMPLE 3D point cloud file [[xyz file](https://github.com/RyuZhihao123/SVDTree/blob/main/Tree1_input.xyz)]** to quickly have a try with our software.


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

