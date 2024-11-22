# SVDTree.

This repo includes the program and software for our paper **SVDTree (Accepted by CVPR 2024)**.

## Project Hierarchy:
This project was led by two co-first authors, each responsible for different parts of the project:

- [Zhihao Liu](https://ryuzhihao123.github.io/): handles the **``Shape-guided 3D Tree Modeling``**, which takes **Voxels (Point Clouds)** as input and produces the final realistic 3D tree models. Please refer to **\[Sec.1\]** for more details
![image info](https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_0.png)

- [Yuan Li](): handles the **``Voxel Diffusion Network``**, which generate rough voxels from tree images.
![image info](https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_1.png)

## \[Section 1\] Shape-guided 3D Tree Reconstruction. 

In this section, we released a **more generic version** of our codes so that you can easily handle both low-precision voxels or precise point clouds. 

### Resources, codes, and demo:


📺 **Demo Video:** We strongly suggest first watching this **[[Demo Video]](https://drive.google.com/file/d/1htelf6xldyFYocqnZ6rtEZxSvwj3Gy1I/view?usp=sharing)** to see the intructions of the usage of my software.

🟥 \*\* **Software Download** \*\*: This is the compiled exe software that you can directly use without any configuration. Please download the entire **[[This Folder]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe)** to get the software. After downloading the folder, you can directly run the program by simply double clicking the **``TreeFromPoint.exe``**. 

 (Windowsシステムを使用する場合、このexeファイルをおすすめします！設定やコンパイルは必要ありません！)

📁 \*\* **Source Codes** \*\*: Please check this [**[Folder](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes)**] to download my source codes.



🟦 **Test Data of a 3D Point Cloud:** Please also download **this EXAMPLE 3D point cloud file [[xyz file](https://github.com/RyuZhihao123/SVDTree/blob/main/Tree1_input.xyz)]** to quickly have a try with our software.




<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_1.png" width = "200" alt="ack" title="dasdasdsa title" align=center />
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_2.png" width = "200" alt="ack" title="dasdasdsa title" align=center />
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_3.png" width = "200" alt="ack" title="dasdasdsa title" align=center />
<br/><center><b>Fig. 1. The workflow of our software, <br>please refer to the video for more details.</b></center>
</div>


### Compile the source codes:
The **[codes](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes)** of this part is entirely developed by C++, so you need to install necessary IDE if you want to compile from scratch:

• **IDE:** The IDE I used is [Qt 5.8](https://download.qt.io/new_archive/qt/5.8/5.8.0/) (qt-opensource-windows-x86-mingw530-5.8.0.exe). I highly recommend you to use Qt as well since you may skip many unexpected compile errors. 
Although visual studio is also an alternative, I haven't physically tested on other C++ IDEs. 

• **Compile:** You can directly compile in Qt Creator by pressing **``CTRL+R``**, or simply run the following script in command line:

```cpp
cd ./TreeFromPoints_codes
qmake TreeFromPoints.pro
```

Then, please check the newly-compiled exe file under **``./release/``** path, and you can freely run it by clicking it.


## \[Section 2\] Voxel Diffusion Network. 

In this section, we released the **neural network** that we design for voxel inference. 

### Source codes:
The network is implemented in Python and PyTorch, please download the codes from this [folder](https://github.com/RyuZhihao123/SVDTree/tree/main/DiffTreeVxl)**)