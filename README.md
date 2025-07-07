# SpectroEase — A Visual Workflow for Spectral Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **SpectroEase** is an open-source, extensible desktop application.
> It converts the entire pipeline—**data loading /partitioning → preprocessing → feature selection → modelling (with hyper-parameter optimization) → evaluation → reporting**—into a drag-and-drop visual workflow, enabling scientists and engineers to perform qualitative or quantitative spectral analysis with zero coding.

---

<details>
<summary><strong>Table of Contents</strong> (click to expand)</summary>

1. [Key Features](#key-features)
2. [UI Overview](#ui-overview)
3. [System Architecture](#system-architecture)
4. [Algorithm Catalogue](#algorithm-catalogue)
5. [System Requirements](#system-requirements)
6. [Dependencies](#dependencies)
7. [Installation & Quick Start](#installation--quick-start)
8. [Data Format](#data-format)
9. [Performance Notes](#performance-notes)
10. [Licence & Disclaimer](#licence--disclaimer)

</details>

---

## Key Features

**​​Multi-format Data Import**​​​​: CSV (optimal performance) / TXT / Excel formats with automated label recognition and data validation
**​​​​Advanced Preprocessing**​​​​: Baseline correction, smoothing, scatter correction, normalization, derivative transformation, peak alignment, and other mainstream preprocessing techniques
**​​​​Feature Selection**​​​​: Principal Component Analysis (PCA), Partial Least Squares Regression (PLSR), wavelet transform, automated peak identification, and established feature selection methodologies
**​​​​Modeling**​​​​: Native support for >15 qualitative and quantitative chemometric algorithms
**​​​​Hyper-parameter Optimization**​​​​: Grid search / Random search / Genetic algorithm optimization strategies
**​​​​Visual Diagnostic Evaluation**​​​​: ROC curves, confusion matrices, feature importance metrics, regression residual analysis with associated quantitative metrics and graphical representations
**​​​​Workflow Interface Design​​**​​: Hierarchical workflow architecture with parametric control panels to reduce learning thresholds
---

## UI Overview


|             Main Window             | 
| :---------------------------------: | 
|![image](https://github.com/user-attachments/assets/b8639d31-b265-4200-8c27-c935ab65daed)
|            Pipeline                 | 
| ![image](https://github.com/user-attachments/assets/0e819ef5-3819-43ba-987c-0b36abe8f739)

---

## System Architecture


```text
SpectroEase/
├── app/                    
│   ├── main.py             
│   ├── gui/                
│   ├── core/               
│   └── utils/             
├── plugins/               
│   ├── preprocessing/    
│   ├── feature_selection/ 
│   ├── modeling/         
│   └── data_partition/     
├── interfaces/         
│   ├── base_plugin.py
│   ├── preprocessing_interface.py
│   ├── feature_selection_interface.py
│   ├── modeling_interface.py
│   └── data_partition_interface.py
└── config/                
```

---

## Algorithm Catalogue

### 1 · Data Partitioning

* Train–Test Split
* K-Fold / Stratified K-Fold
* Leave-One-Group-Out (LOGO)
* Random Split

### 2 · Pre-processing

| Category                 | Algorithms                                                                   |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Baseline / Smoothing** | Baseline Correction · Savitzky–Golay · Moving Average · Median · Gaussian    |
| **Scatter Correction**   | SNV · MSC · EMSC · RNV · OSC                                                 |
| **Normalisation**        | Standard Scale · Min–Max · Vector · Area · Maximum                           |
| **Derivatives**          | First · Second · Savitzky–Golay Derivative · Finite Difference · Gap-Segment |
| **Peak Alignment**       | DTW · COW · ICS · PAFFT                                                      |
| **Other Tools**          | SNV Processor · Spectrum Converter · Spectrum Visualizer · Custom Pipeline   |

### 3 · Feature Selection

* PCA · PLSR · Automatic Peak Detection · Wavelet Transform

### 4 · Modelling

#### 4.1 Qualitative (Classification / Clustering)

Logistic Regression · LDA / QDA · SVM · KNN · Decision Tree · Random Forest · Gradient Boosting · XGBoost · Neural Network · K-Means · Hierarchical · DBSCAN

#### 4.2 Quantitative (Regression)

MLR · PLSR · SVR · Decision Tree Regression · Random Forest Regression · GPR · Ridge · Lasso · ElasticNet

### 5 · Hyper-parameter optimization

Grid Search · Random Search · Genetic Algorithm

---

## System Requirements

| Component  | Specification                            |
| ---------- | ---------------------------------------- |
| **OS**     | Windows 10 / 11 (64-bit)                 |
| **Python** | ³ ≥ 3.8 (recommended 3.11)               |
| **RAM**    | ≥ 4 GB (recommended 8 GB +)              |
| **Disk**   | ≥ 2 GB free (recommended 5 GB +)         |
| **GPU**    | Not required—current release is CPU-only |

---

## Dependencies

The platform is written in **Python 3.11.9** and relies on the following libraries:

| Library      | Version tested | Purpose                        |
| ------------ | -------------- | ------------------------------ |
| NumPy        | 1.26.4         | Numerical computing            |
| pandas       | 1.5.3          | Data manipulation              |
| SciPy        | 1.11.1         | Scientific algorithms          |
| Matplotlib   | 3.10.0         | Visualisation                  |
| scikit-learn | 1.2.2          | Machine-learning workflows     |
| PyQt5        | 5.15.11        | Graphical user interface       |
| ReportLab    | 4.3.1          | PDF / report generation        |
| OpenPyXL     | 3.1.5          | Spreadsheet interoperability   |
| onnx         | ≥ 1.12.0       | Model export                   |
| skl2onnx     | ≥ 1.14.0       | scikit-learn → ONNX conversion |

All versions satisfy SpectroEase’s compatibility requirements; newer versions should work but are not actively tested.

---

## Installation & Quick Start

### 1 · Clone the Repo & Install Dependencies

```powershell
git clone [https://github.com/shudayi/SpectroEase-V1.0]
cd SpectroEase
pip install -r requirements.txt      # exact versions are pinned
```

### 2 · Launch the Application

```powershell
python main.py                       # start the GUI
```

### 3 · Typical Workflow

1. **Import / Split** → 2. **Pre-process** → 3. **Select Features**
2. **Model / Tune** → 5. **Visualise & Evaluate** → 6. **Export Report**

---

## Data Format

| Aspect                 | Details                                         |
| ---------------------- | ----------------------------------------------- |
| **Supported files**    | CSV · TXT · Excel                               |
| **Recommended layout** | Row-wise: `Sample_ID, Label, 400 nm, 402 nm, …` |
| **Label keywords**     | `category · class · label · variety · target`   |
| **Demo dataset**       | `datasets/seed_demo/` (CC-BY-4.0)               |

![image](https://github.com/user-attachments/assets/de2a0556-5729-48fc-b10f-c6efb85da488)






---

## Performance Notes

* Multi-threaded UI keeps the interface responsive
* CPU multi-processing for heavy computation
* Smart cache automatically re-uses identical intermediate results

---



## Licence & Disclaimer

* **Code** — MIT Licence (see `LICENSE.txt`)
* **Sample data** — CC-BY-4.0
* **Disclaimer** — This software is provided **for research and educational purposes only**.
  The authors accept no liability for commercial use or any losses arising from model mis-predictions.
  Users must comply with local laws and regulations.

---

**SpectroEase — Making Spectral Analysis Simple**

<hr>

# SpectroEase：可视化光谱分析全流程软件

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **SpectroEase** 是一款开源、可扩展的可视化应用程序。
> 它将光谱分析环节全部流程： **数据载入/分割 → 预处理 → 特征选择 → 建模（含超参数优化） → 评估 → 报告** 封装为可视模块化工作流，帮助科研人员和工程师零代码完成光谱定性/定量分析。

---

<details>
<summary><strong>目录（点击展开）</strong></summary>

1. [技术特点](#技术特点)
2. [界面概览](#界面概览)
3. [系统架构](#系统架构)
4. [完整算法清单](#完整算法清单)
5. [系统需求](#系统需求)
6. [依赖库](#依赖库)
7. [安装与快速上手](#安装与快速上手)
8. [数据格式说明](#数据格式说明)
9. [性能优化](#性能优化)
10. [许可证与免责声明](#许可证与免责声明)

</details>

---

## 关键特性

* **多格式导入**：CSV / TXT / Excel，自动标签识别与数据校验，使用csv文件效果最佳
* **高级预处理**：基线校正、平滑、散射校正、归一化、导数、波峰对齐等主流预处理方法
* **特征选择**：PCA、PLSR、小波、自动峰检等主流特征选择方法
* **建模算法**：内置定性、定量两大类超15种算法模型
* **超参数优化**：网格搜索 / 随机搜索 / 遗传算法
* **可视化评估**：ROC、混淆矩阵、特征重要性、回归残差等分析相关指标、图表
* **界面模式**：从上到下工作流 + 参数面板，降低上手门槛

---

## 界面概览



|               主界面               |                   预处理流程                   |
| :-----------------------------: | :---------------------------------------: |
| 如英文区 |

---

## 系统架构

```text
SpectroEase/
├── app/                    
│   ├── main.py             
│   ├── gui/                
│   ├── core/               
│   └── utils/              
├── plugins/               
│   ├── preprocessing/     
│   ├── feature_selection/  
│   ├── modeling/         
│   └── data_partition/    
├── interfaces/            
│   ├── base_plugin.py
│   ├── preprocessing_interface.py
│   ├── feature_selection_interface.py
│   ├── modeling_interface.py
│   └── data_partition_interface.py
└── config/               
```


---

## 完整算法清单

### 1 · 数据划分

* Train–Test Split
* K-Fold / Stratified K-Fold
* Leave-One-Group-Out (LOGO)
* Random Split

### 2 · 预处理

| 分类          | 算法                                                                           |
| ----------- | ---------------------------------------------------------------------------- |
| **基线 / 平滑** | Baseline Correction · Savitzky-Golay · Moving Average · Median · Gaussian    |
| **散射校正**    | SNV · MSC · EMSC · RNV · OSC                                                 |
| **归一化**     | Standard Scale · Min-Max · Vector · Area · Maximum                           |
| **导数**      | First · Second · Savitzky-Golay Derivative · Finite Difference · Gap-Segment |
| **波峰对齐**    | DTW · COW · ICS · PAFFT                                                      |
| **其他工具**    | SNV Processor · Spectrum Converter · Spectrum Visualizer · 自定义流水线            |

### 3 · 特征选择

* PCA · PLSR · 自动峰检 · Wavelet Transform 等

### 4 · 建模

#### 4.1 定性分析

Logistic Regression · LDA / QDA · SVM · KNN · Decision Tree · Random Forest · Gradient Boosting · XGBoost · Neural Network · K-Means · Hierarchical · DBSCAN 等

#### 4.2 定量分析

MLR · PLSR · SVR · Decision Tree Regression · Random Forest Regression · GPR · Ridge · Lasso · ElasticNet等

### 5 · 超参数优化

Grid Search · Random Search · Genetic Algorithm

---

## 系统需求

| 项目         | 规格                       |
| ---------- | ------------------------ |
| **操作系统**   | Windows 10 / 11 (64-bit) |
| **Python** | ≥ 3.8（推荐 3.11）           |
| **内存**     | ≥ 4 GB（推荐 8 GB 以上）       |
| **磁盘空间**   | ≥ 2 GB（推荐 5 GB 以上）       |
| **GPU**    | 无硬性要求，当前版本仅使用 CPU        |

---

## 依赖库

本平台基于 **Python 3.11.9** 开发，主要依赖如下：

| 库            | 测试版本     | 作用                     |
| ------------ | -------- | ---------------------- |
| NumPy        | 1.26.4   | 数值计算                   |
| pandas       | 1.5.3    | 数据处理                   |
| SciPy        | 1.11.1   | 科学计算                   |
| Matplotlib   | 3.10.0   | 可视化                    |
| scikit-learn | 1.2.2    | 机器学习框架                 |
| PyQt5        | 5.15.11  | 图形界面                   |
| ReportLab    | 4.3.1    | PDF / 报告生成             |
| OpenPyXL     | 3.1.5    | Excel 读写               |
| onnx         | ≥ 1.12.0 | 模型导出                   |
| skl2onnx     | ≥ 1.14.0 | scikit-learn → ONNX 转换 |

上述版本均通过兼容性测试；更新版本一般亦可正常运行，但未做长期验证。

---

## 安装与快速上手

### 1 · 克隆源码并安装依赖

```powershell
git clone https://github.com/shudayi/SpectroEase-V1.0
cd SpectroEase
pip install -r requirements.txt   # 已固定精确版本
```

### 2 · 启动应用

```powershell
python main.py                    # 打开 GUI
# 或运行批处理工作流示例
python main.py --workflow examples/pca_svm.yml
```

### 3 · 基础工作流

1. **导入数据 / 分割** → 2. **选择预处理方法** → 3. **特征选择**
2. **建模 / 超参优化** → 5. **评估可视化** → 6. **导出报告**
| ![image](https://github.com/user-attachments/assets/0e819ef5-3819-43ba-987c-0b36abe8f739)
---

## 数据格式说明

| 维度        | 说明                                          |
| --------- | ------------------------------------------- |
| **支持格式**  | CSV · TXT · Excel                           |
| **推荐布局**  | 行式：`Sample_ID, Label, 400nm, 402nm, …`      |
| **标签关键字** | category · class · label · variety · target |
| **示例数据集** | `datasets/` (CC-BY-4.0)           |


![image](https://github.com/user-attachments/assets/7c84d14b-e3d1-478e-a1b0-3117c9c72e4a)


---






---

## 许可证与免责声明

* **代码**：MIT License（见 `LICENSE.txt`）
* **示例数据**：CC-BY-4.0
* **免责声明**：本软件仅供科研与教学使用。作者不对商业用途或模型误判造成的任何损失负责；使用者需遵守当地法律法规。

---

**SpectroEase —— 让光谱分析变简单**
