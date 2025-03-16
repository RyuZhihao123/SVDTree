# SVDTree.

This page includes the program and software toolkits for our paper **SVDTree (Accepted by CVPR 2024)**.

## Project Hierarchy:
This project was led by two co-first authors, each responsible for different parts seperately:

🐶 [**Zhihao Liu**](https://ryuzhihao123.github.io/): mainly handles **``Shape-guided 3D Tree Modeling``** (3D Graphics part), which takes **voxels (point clouds)** as input and produces the final realistic 3D tree models. 

• [[Code (C++)]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes) [[Demo Video]](https://drive.google.com/file/d/1htelf6xldyFYocqnZ6rtEZxSvwj3Gy1I/view?usp=sharing) [[EXE Program]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe). Please refer to Sec.1 for detailed introduction.

• ⭐ Please also refer to my another repository for a [more advanced point cloud 3D tree reconstruction software](https://github.com/RyuZhihao123/Point-cloud-3D-tree-reconstruction).

<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_0.png" height = "150" alt="ack" title="dasdasdsa title" align=center />
</div>

🐶 [**Yuan Li**](): mainly handles the **``Voxel Diffusion Network``** (CNN part), which is trained to generate rough semantic voxels.

• [[Code (Python)]](https://github.com/RyuZhihao123/SVDTree/tree/main/DiffTreeVxl) Please refer to Sec.2 for detailed introduction.

<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_1.png" height = "150" alt="ack" title="dasdasdsa title" align=center />
</div>


## Section 1: Shape-guided 3D Tree Reconstruction. 

In this section, we released a more adaptable version of our code, that is capable of reconstructing 3D tree structures from both **``voxels``** or **``dense point clouds``**. Typically, handling point clouds poses greater challenges compared to voxels, which is our default input setting. We hope this version could be helpful if you need to process such complex data format.

Should you have any questions on this part, please feel free to reach out to [**Zhihao Liu**](https://ryuzhihao123.github.io/).


### 1.1 - Resources, codes, and demo:


📺 **(1) Demo Video:**

 We strongly recommend watching this [[Demo Video]](https://drive.google.com/file/d/1htelf6xldyFYocqnZ6rtEZxSvwj3Gy1I/view?usp=sharing) to quickly know about the basic usage of the software.


🟥 **(2) Software Download**: 

We have released a compiled exe software that can be directly used without any configuration on Windows PCs. Please download the entire [[this Folder]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_exe) to access the software. Once downloaded, you can easily run the program by simply double-clicking the **``TreeFromPoint.exe``**. 


📁 **(3) Source Code**: 

Please check this [[Folder]](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes) to download the source code.



🟦 **(4) Test Data of a 3D Point Cloud:**

 You can download **this example 3D point cloud file [[xyz file](https://github.com/RyuZhihao123/SVDTree/blob/main/Tree1_input.xyz)]** to quickly have a try with our software. If you would like to generate trees using your own point data, please prepare the file with reference to this format.




<div align=center>
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_1.png" height = "150" alt="ack" title="dasdasdsa title" align=center />
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_2.png" height = "150" alt="ack" title="dasdasdsa title" align=center />
<img src="https://github.com/RyuZhihao123/SVDTree/blob/main/Fig_UI_3.png" height = "150" alt="ack" title="dasdasdsa title" align=center />
<br/><center><b>Fig. 1. The workflow of our software, <br>please refer to the video for more details.</b></center>
</div>


### 1.2 - Compile the Source Code:
The **[code](https://github.com/RyuZhihao123/SVDTree/tree/main/TreeFromPoints_codes)** for Shape-driven Tree modeling is entirely developed by C++, so you'll need to install the necessary IDE if you want to compile it from scratch.

📁 **IDE:**

The IDE I used for programming is [**Qt**](https://download.qt.io/new_archive/qt/5.8/5.8.0/) (qt-opensource-windows-x86-mingw530-5.8.0.exe). 

I highly recommend you to use it as well, since I have successfully compiled my code with it, which can help you avoid many unexpected compilation errors.
Certainly, other IDEs like Visual Studio, are also alternatives.

📁 **Compile:** 

You can directly compile in Qt by pressing **``CTRL+R``** after opening the project,

or simply run the following script in the command line:

```cpp
cd ./TreeFromPoints_codes
qmake TreeFromPoints.pro
```

Then, you can find the newly-compiled exe program under path **``./release/``**. and you can activate it easily by clicking it.


## Section 2: Voxel Diffusion Network. 

In this section, we released the **neural network** that is designed for semantic voxel inference.  

Should you have any questions on this part, please feel free to reach out to [**Yuan Li**]().

### 2.1 - Installation:
• **Download:** 

Please download the code for this part from this [[Folder]](https://github.com/RyuZhihao123/SVDTree/tree/main/DiffTreeVxl).

• **Dependencies:**

```
conda create -n DiffTreeVxl
conda activate DiffTreeVxl

conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=10.2 -c pytorch -c nvidia

pip install tqdm fire einops pyrender pyrr trimesh ocnn timm scikit-image==0.18.2 scikit-learn==0.24.2 pytorch-lightning==1.6.1
```

• **Data Preparation:**

We prepare the dataset of diverse 3D tree models using a software that was developed by [**Zhihao Liu**](https://ryuzhihao123.github.io/). Please refer to this [**[GitHub page]**](https://github.com/RyuZhihao123/CHITreeProject?tab=readme-ov-file#1-tree-dataset-generator) for more details and to download the generation tool.


Notebly, our tool enables automatic generation of realistic 3D trees with diverse species, and has been used for generating dataset by many external labs in their projects, including several SIGGRAPH/CHI/CVPR papers (e.g., [TreePartNet](https://vcc.tech/research/2021/TreePartNet)).




### 2.2 - Usage:
• **Training:** 
```
python train.py --config ./config/config.yaml
```

You can freely change the training settings by customizing the yaml file.

• **Test:** 
```
python predict.py
```

In predict.py, You can set the paths to ckpt and yaml files by changing the Line 9 and Line 12, respectively.


