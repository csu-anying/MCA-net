# MCA-net: A multi-task channel attention network for Myocardial infarction detection and location using 12-lead ECGs
MCA-net工作流的功能如下：
- 利用患者10s的12导联的心电信号数据预测是否存在心梗，以及对心梗的位置进行定位。


#### 详情请见以下文章
> [**MCA-net: A multi-task channel attention network for Myocardial infarction detection and location using 12-lead ECGs**](https://www.sciencedirect.com/science/article/pii/S0010482522009076)<br/>
Weibai Pan, Ying An, Yuxia Guan, Jianxin Wang,
MCA-net: A multi-task channel attention network for Myocardial infarction detection and location using 12-lead ECGs,
Computers in Biology and Medicine,
Volume 150,
2022,
106199,
ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2022.106199.


## 输入数据要求

- 格式要求：10s的12导联的心电信号，是一个 12 X 5000 (或者 5000 X 12)维度的数据；
  
- 命名要求：以可以区分样本的ID命名，并且以“.npy”为后缀;
- 路径要求：放在data/user_predict下，如data/user_predict/xxx.npy。

## 输出数据

- 检测分类：检测是否存在心肌梗塞(Myocardial Infarction，MI)，是一个二分类任务，如果存在，输出MI，不存在输出NORM;
  
- 定位分类-5：如果存在心梗，则定位心梗的位置，是一个五分类任务，分别为AMI(Anterior MI), ASMI(Anteroseptal MI), ALMI(Anterolateral MI), other(other location MI), NORM(Healthy);
  
- 定位分类-7：如果存在心梗，则定位心梗的位置，是一个七分类任务，分别为AMI(Anterior MI), ASMI(Anteroseptal MI), ALMI(Anterolateral MI), IMI(Inferior MI), ILMI(Inferolateral MI), other(other location MI), NORM(Healthy).

## 安装
通过Conda使用如下命令安装environment.yaml文件中的依赖

`conda create -f environment.yaml`

## 使用

- 使用测试脚本测试
  - 激活安装好的环境
   
    `conda activate environment` 

  - 移动到项目根目录
  
    `cd WorkFlow_PWB`
  
  - 使用next flow命令调用
    
    `nextflow run main.nf --mode val`

  `结果文件保存在项目根目录下的result/valid_evaluation.txt文件中`

- 使用模型进行预测
  - 激活安装好的环境
   
    `conda activate environment` 

  - 移动到项目根目录
        
    `cd WorkFlow_PWB`
  
  - 使用next flow命令调用
  
    `nextflow run main.nf --mode predict`
  

  `结果文件保存在项目根目录下的result/user_predict_results.txt文件中`


## 项目文件结构

```html
WorkFlow_PWB:
├─data（存放训练数据和测试数据）
├─result（结果文件夹）
├─saved_models（模型参数文件夹）
└─__pycache__
```
