# SVDTree.

This repo includes the program of our paper **SVDTree (Accepted by CVPR 2024)**.
This work was conducted one year before I joined UTokyo.

- I am one of the joint first authors ([Zhihao Liu](https://ryuzhihao123.github.io/)), who take charge of the implementation of the **second main step** of this paper, i.e. **``Voxel (Point Cloud)-based 3D Tree Reconstruction``**.
This part is entirely implemented by myself so I released it seperately in my own Github.
 

![image info](https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_0.png)


## Point Cloud (Voxel)-based 3D Tree Reconstruction. 

Since point clouds are way more complex and common than the voxel representation used in our paper, we released a **more Generic Version** of our codes so that you can easily handle both two data representations more robustly. 

- 📺 **Demo Video:** Please first watch this **[[Demo Video]](https://drive.google.com/file/d/1htelf6xldyFYocqnZ6rtEZxSvwj3Gy1I/view?usp=sharing)** to see the intructions of the usage of my software.

- 🟥 **Download link:** Please get our software by downloading the entire **[[This Folder]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe)**. If you're using **Win10+**, we strongly suggest you to try **the EXE version** that we released for quick start . After downloading the entire [directory](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe), you can directly run the program by double clicking the **``TreeFromPoint.exe``**. (Windowsシステムを使用する場合、このexeファイルをおすすめします！設定やコンパイルは必要ありません！)

- 📁 **Codes**: Please check this [**[Folder](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes)**] to download my source codes.

- 🟦 **Test Data of a 3D Point Cloud:** Please also download **this EXAMPLE 3D point cloud file [[xyz file](https://github.com/RyuZhihao123/SVDTree/blob/main/Tree1_input.xyz)]** to quickly have a try with our software.


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

