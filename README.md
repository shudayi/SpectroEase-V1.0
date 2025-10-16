# SpectroEase — A Visual Workflow for Spectral Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **SpectroEase** is an open-source, extensible desktop application.
> It converts the entire pipeline—**data loading and partitioning → preprocessing → feature selection → modeling (with hyper-parameter optimization) → evaluation → reporting**—into a drag-and-drop visual workflow, enabling scientists and engineers to perform qualitative or quantitative spectral analysis with zero coding.

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
8. [Typical Workflow](#typical-workflow)  
9. [Data Format](#data-format)  
10. [Performance Notes](#performance-notes)  
11. [License & Disclaimer](#license--disclaimer)

</details>

---

## Key Features

- **Module architecture aligned with the paper**: five core modules — **data loading and partitioning**, **preprocessing**, **feature selection**, **modeling**, and **evaluation** — each embedding mainstream algorithmic implementations.
- **Task detector**: inspects the target variable to automatically initiate the **classification** (qualitative) or **regression** (quantitative) procedure.
- **Shared in-memory buffer**: passes intermediate results between modules to simplify hand-offs while preserving module independence.
- **Multi-format data import**: CSV (best performance) / TXT / Excel with automated header/type detection, label recognition, and data validation.
- **Advanced preprocessing**: baseline correction, smoothing, scatter correction, normalization, **derivative transforms**, **peak alignment**, **despiking (Raman)**, and **outlier detection**.
- **Feature selection**: PCA, PLSR, wavelet transform, automated peak detection, and established feature-selection methodologies.
- **Modeling**: parallel suites for **classification** and **regression** with sensible defaults and full parameter control.
- **Hyper-parameter optimization**: **grid search**, **randomized search**, and **Bayesian optimization (Tree-structured Parzen Estimator, TPE)** with stratified k-fold (default k = 5) and parallel execution.
- **Evaluation & visualization**: ROC, precision–recall analysis, confusion matrix, feature-importance graphs, regression residuals, and comprehensive metrics.
- **Reporting & export**: publication-ready **PDF** reports; results exportable as **Excel/CSV**; validated models serializable to **Pickle (.pkl)** or **ONNX (.onnx)** for rapid prototyping and edge deployment.
- **GUI for non-programmers**: desktop interface built with PyQt5; Windows standalone executable provided; source build available for Python users.

---

## UI Overview

|               Main Window               |                Pipeline                |
| :-------------------------------------: | :------------------------------------: |
| ![image](https://github.com/user-attachments/assets/b8639d31-b265-4200-8c27-c935ab65daed) | ![image](https://github.com/user-attachments/assets/0e819ef5-3819-43ba-987c-0b36abe8f739) |

---

## System Architecture

```text
SpectroEase/
├── main.py                 # Application entry point
├── app/                    # Main application
│   ├── controllers/        # Controller layer
│   ├── models/             # Data models
│   ├── services/           # Business logic services
│   ├── views/              # User interface
│   └── utils/              # Utility functions
├── plugins/                # Plugin modules
│   ├── data_partitioning/  # Data partitioning algorithms
│   ├── preprocessing/      # Preprocessinging algorithms
│   ├── feature_selection/  # Feature selection methods
│   ├── modeling/           # Machine learning models
│   └── reporting/          # Report generation
├── interfaces/             # Interface definitions
├── config/                 # Configuration files
├── examples/               # Example datasets
└── translations/           # Multi-language support
```

**Implementation and platform**: SpectroEase is implemented in **Python 3.11.9** using scientific libraries (NumPy, pandas, SciPy, scikit-learn, Matplotlib) with a desktop GUI built on **PyQt5**. The current distribution targets **Windows**: a **standalone executable (.exe)** bundles all dependencies for double‑click use, while a source build remains available for Python users.

**Workflow orchestration**: Operators progress systematically through **data loading and partitioning → preprocessing → feature selection → model selection (with hyper‑parameter optimization) → evaluation**. Intermediate results are passed via a **shared in‑memory buffer**. A **task detector** examines the target variable to determine qualitative vs quantitative analysis and triggers the corresponding pipeline. The system can store validated models locally and optionally generate standardized **PDF** reports that document datasets, algorithms, hyper‑parameters, metrics, and figures.

---

## Algorithm Catalogue

### 1 · Data loading & partitioning

- File import: CSV, TXT, Excel; auto detection of file type & header layout; numeric parsing with full precision; wavelength‑axis extraction and metadata retention.
- Robust handling of missing values and mixed types (conservative casting; warnings surfaced in UI).
- Built‑in **partitioning utilities**: stratified **train–validation–test** splits, K‑Fold, Stratified K‑Fold, LOGO, Random partitioning. Users configure partitioning ratios/folds in the GUI; partitions persist alongside raw data.

### 2 · Preprocessinging

| Category                   | Algorithms / Options                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------ |
| **Despiking (Raman)**      | Median Absolute Deviation (**MAD**), **local z‑score** with tunable window/threshold |
| **Baseline Correction**    | Polynomial, **ALS**, **airPLS**                                                      |
| **Scatter Correction**     | **SNV**, **MSC**, **EMSC**, **RNV**, **OSC**                                         |
| **Smoothing**              | **Savitzky–Golay**, Moving Average, Median Filter, Gaussian, Wavelet                 |
| **Scaling & Enhancement**  | Standard Scale, Min–Max Scale, L2 Normalize, Vector, Area, Maximum, **First/Second Derivative**, Savitzky–Golay Derivative, Finite Difference, Gap‑Segment, **Peak Alignment**, **Outlier Detection** |

### 3 · Feature Selection

- **Transform methods**: **PCA**, **Wavelet Transform**
- **Model‑based**: **PLSR**
- **Spectroscopy‑specific**: **Peak detection**

### 4 · Modeling

**Classification (qualitative)**: **Logistic Regression**, **SVM** (linear/polynomial/RBF/sigmoid), **KNN**, **Decision Tree**, **Random Forest**, **Gradient Boosting ensembles** (e.g., XGBoost/LightGBM), **MLP neural networks**.  
**Regression (quantitative)**: **Multiple Linear Regression (MLR)**, **PLSR**, **SVR**, **Decision‑Tree Regressor**, **Random‑Forest Regressor**, **Ridge**, **LASSO**.  
All estimators expose tunable hyper‑parameters (e.g., regularization strength, kernel type, tree depth, learning‑rate schedule).

### 5 · Hyper‑parameter Optimization

- **Grid search** (exhaustive), **Randomized search** (stochastic), and **Bayesian optimization (TPE)** concentrate evaluations on high‑potential regions.  
- Shared **stratified k‑fold** backbone (default **k = 5**), parallel execution via `concurrent.futures`, and full trial logging for reproducibility.  
- The **best configuration** is retrained on the combined **train + validation** set before hand‑off to evaluation.

### 6 · Evaluation

- **Classification**: accuracy, precision, recall, **F1‑score**, **confusion matrix**.  
- **Regression**: **R²**, **MSE**, **MAE**, and additional regression indicators; optional confidence‑band estimation for deployment.  
- **Visualization**: **ROC curves**, **precision–recall** analysis, class‑distribution charts, **feature‑importance** graphs, **regression residuals**; all graphics can be embedded automatically in the final **PDF** report.  
- **Export**: results exportable as **Excel/CSV**; validated models serialized as **Pickle (.pkl)** or **ONNX (.onnx)** for rapid prototyping and edge‑device deployment.

---

## System Requirements

| Component  | Specification                            |
| ---------- | ---------------------------------------- |
| **OS**     | Windows 10 / 11 (64-bit)                 |
| **Python** | ≥ 3.8 (recommended 3.11)                 |
| **RAM**    | ≥ 4 GB (8 GB+ recommended)               |
| **Disk**   | ≥ 2 GB free (5 GB+ recommended)          |
| **GPU**    | Not required — current release is CPU-only |

---

## Dependencies

The platform is written in **Python 3.11.9** and relies on the following libraries:

| Library       | Version tested | Purpose                      |
| ------------- | -------------- | ---------------------------- |
| PyQt5         | 5.15.7         | Graphical user interface     |
| NumPy         | 2.3.3          | Numerical computing          |
| pandas        | 2.3.3          | Data manipulation            |
| SciPy         | 1.16.2         | Scientific algorithms        |
| scikit-learn  | 1.7.2          | Machine-learning workflows   |
| Matplotlib    | 3.10.7         | Visualization                |
| seaborn       | 0.13.2         | Statistical visualization    |
| pyqtgraph     | 0.13.7         | Interactive plotting         |
| ReportLab     | 4.4.4          | PDF / report generation      |
| OpenPyXL      | 3.1.5          | Excel file handling          |
| xlrd          | 2.0.2          | Excel file reading           |
| deap          | 1.4.3          | Genetic algorithms           |
| requests      | 2.32.5         | HTTP library                 |
| Pillow        | 11.3.0         | Image processing             |
| joblib        | 1.5.2          | Parallel computing           |
| threadpoolctl | 3.6.0          | Thread pool control          |

**Note**: The table lists the main dependencies. Additional supporting libraries (contourpy, cycler, fonttools, kiwisolver, pyparsing, urllib3, certifi, idna, charset-normalizer, et-xmlfile, packaging, setuptools, six, python-dateutil, pytz, tzdata) are installed automatically as sub-dependencies. All versions satisfy SpectroEase’s compatibility requirements; newer versions should work but are not actively tested.

---

## Installation & Quick Start

### Option 1: For General Users (Recommended)

Two pre-built packages are available for users who prefer not to work with source code.

#### A. Standalone Executable
Download `SpectroEase.exe` from **[Google Drive](https://drive.google.com/file/d/1BvLx0z0h46n3n_obOIHizyThG2UIJ_rh/view?usp=drive_link)**. This single file runs directly with no installation.

#### B. Folder Version
Download `EXE for SpectroEase.zip` from the **[GitHub Releases](https://github.com/shudayi/SpectroEase-V1.0/releases/tag/v1.0.0)** page. Unzip it and run `SpectroEase.exe`. This version may start faster.

### Option 2: For Developers (from Source)

#### 1 · Clone the Repo & Install Dependencies

```powershell
git clone https://github.com/shudayi/SpectroEase-V1.0
cd SpectroEase-V1.0
pip install -r requirements.txt      # exact versions are pinned
```

#### 2 · Launch the Application

```powershell
python main.py                       # start the GUI
```

---

## Typical Workflow

1. **Data loading & partitioning** → 2. **Preprocessinging** → 3. **Feature selection**  
4. **Modeling & hyper-parameter optimization** → 5. **Evaluation & visualization** → 6. **Reporting**

> **Demo video**  
> [Watch on Google Drive »](https://drive.google.com/file/d/1-Q8o-1CNyoxUC4yIulCBptrKXfMi99-s/view?usp=drive_link)

[![Watch the demo (SpectroEase Workflow)](https://drive.google.com/thumbnail?id=1-Q8o-1CNyoxUC4yIulCBptrKXfMi99-s)](https://drive.google.com/file/d/1-Q8o-1CNyoxUC4yIulCBptrKXfMi99-s/view?usp=drive_link)

**Recommended sequence**
1. **Data loading & partitioning** (stratified Train/Validation/Test, K‑Fold, Stratified K‑Fold, LOGO, Random)  
2. **Preprocessinging** (baseline correction, smoothing, scatter correction, normalization, **derivative transforms**, **peak alignment**, **despiking (Raman)**, **outlier detection**)  
3. **Feature selection** (PCA / PLSR / Wavelet / Peak detection)  
4. **Model selection & hyper‑parameter optimization** (classification & regression suites; **Grid / Randomized / Bayesian (TPE)**)  
5. **Evaluation & visualization** (ROC, precision–recall, confusion matrix, feature-importance, regression residuals)  
6. **Reporting** (**PDF**; results to **Excel/CSV**; models as **.pkl**/**.onnx**)

> *Note:* GitHub READMEs cannot embed Google Drive videos for inline playback. The clickable thumbnail above links to the hosted video.

---

## Data Format

| Aspect                 | Details                                         |
| ---------------------- | ----------------------------------------------- |
| **Supported files**    | CSV · TXT · Excel                               |
| **Recommended layout** | Row-wise: `Sample_ID, Label, 400 nm, 402 nm, …` |
| **Label keywords**     | `category · class · label · variety · target`   |
| **Demo dataset**       | `examples/` (CC-BY-4.0)                          |

![image](https://github.com/user-attachments/assets/de2a0556-5729-48fc-b10f-c6efb85da488)

---

## Performance Notes

- Multi-threaded UI keeps the interface responsive.  
- CPU multi-processing accelerates heavy computation.  
- **Shared in‑memory buffer** and smart caching automatically reuse identical intermediate results.

---

## License & Disclaimer

- **Code** — MIT License (see `LICENSE.txt`)  
- **Sample data** — CC-BY-4.0  
- **Disclaimer** — This software is provided **for research and educational purposes only**.  
  The authors accept no liability for commercial use or any losses arising from model mispredictions.  
  Users must comply with local laws and regulations.

---

**SpectroEase — Making Spectral Analysis Simple**

<hr>

# SpectroEase：可视化光谱分析全流程软件

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **SpectroEase** 是一款开源、可扩展的可视化应用程序。
> 它将光谱分析环节全部流程： **1.数据载入/分割   2.预处理   3.特征选择   4.建模（含超参数优化）  5.评估  6.报告** 打包为可视模块化工作流，帮助科研人员和工程师零代码完成光谱定性/定量分析。

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
* **预处理**：基线校正、平滑、散射校正、归一化、导数、波峰对齐等主流预处理方法
* **特征选择**：PCA、PLSR、小波、自动峰检等主流特征选择方法
* **建模算法**：内置定性、定量两大类超15种算法模型
* **超参数优化**：网格搜索 / 随机搜索 / 遗传算法
* **可视化评估**：ROC、混淆矩阵、特征重要性、回归残差等分析相关指标、图表
* **界面模式**：从上到下工作流 + 参数面板，降低上手门槛

---

## 界面概览



|             主界面             |                   流程图                   |
| :-----------------------------: | :---------------------------------------: |
|![image](<img width="1777" height="1136" alt="image" src="https://github.com/user-attachments/assets/cbfeb28e-5399-4d28-ac9b-56d10c885898" />)|![image](![论文图片_05](https://github.com/user-attachments/assets/a6a7c7ee-86d3-41a5-bd8f-f4ffd64b0c42)
)|

---

## 系统架构

```text
SpectroEase/
├── main.py                 # 应用程序入口
├── app/                    # 主应用程序
│   ├── controllers/        # 控制器层
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑服务
│   ├── views/             # 用户界面
│   └── utils/             # 工具函数
├── plugins/               # 插件模块
│   ├── data_partitioning/ # 数据分割算法
│   ├── preprocessing/     # 预处理算法
│   ├── feature_selection/ # 特征选择方法
│   ├── modeling/          # 机器学习模型
│   └── reporting/         # 报告生成
├── interfaces/            # 接口定义
├── config/                # 配置文件
├── examples/              # 示例数据集
└── translations/          # 多语言支持
```


---

## 完整算法清单

### 1 · 数据划分

* Train–Test Split
* K-Fold Cross-Validation
* Stratified K-Fold
* Leave-One-Group-Out (LOGO)
* Random Split

### 2 · 预处理

| 分类          | 算法                                                                           |
| ----------- | ---------------------------------------------------------------------------- |
| **去峰处理**    | MAD (中值绝对偏差) · Local Z-score (局部Z分数)                                      |
| **基线校正**    | Polynomial (多项式) · ALS (非对称最小二乘) · airPLS                               |
| **散射校正**    | SNV · MSC · EMSC · RNV · OSC                                                 |
| **平滑处理**    | Savitzky-Golay · Moving Average · Median Filter · Gaussian · Wavelet         |
| **缩放与增强**  | Standard Scale · Min-Max Scale · L2 Normalize · Vector · Area · Maximum · First Derivative · Second Derivative · Savitzky-Golay Derivative · Finite Difference · Gap-Segment · Denoising · Peak Alignment · Outlier Detection |

### 3 · 特征选择

| 分类          | 算法                                                                           |
| ----------- | ---------------------------------------------------------------------------- |
| **变换方法**    | PCA (主成分分析) · Wavelet Transform (小波变换)                                  |
| **模型方法**    | PLSR (偏最小二乘回归)                                                        |
| **光谱专用**    | Peak Detection (寻峰)                                                        |

### 4 · 建模

#### 4.1 定性分析 (Classification)

SVM · 随机森林 (RF) · K-近邻 (KNN) · 决策树 (DT) · 神经网络 (NN) · XGBoost · LightGBM

#### 4.2 定量分析 (Regression)

偏最小二乘回归 (PLSR) · 支持向量回归 (SVR) · 随机森林 (RF) · 神经网络 (NN) · 高斯过程回归 (GPR) · XGBoost · LightGBM

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

| 库           | 测试版本  | 
| ------------ | -------- | 
| PyQt5        | 5.15.7   |
| NumPy        | 2.3.3    | 
| pandas       | 2.3.3    | 
| SciPy        | 1.16.2   | 
| scikit-learn | 1.7.2    | 
| Matplotlib   | 3.10.7   | 
| seaborn      | 0.13.2   | 
| pyqtgraph    | 0.13.7   | 
| ReportLab    | 4.4.4    | 
| OpenPyXL     | 3.1.5    |
| xlrd         | 2.0.2    | 
| deap         | 1.4.3    | 
| requests     | 2.32.5   | 
| Pillow       | 11.3.0   | 
| joblib       | 1.5.2    | 
| threadpoolctl| 3.6.0    | 

**说明**：上表列出了主要依赖库。其他支持库作为子依赖会自动安装。上述版本均通过兼容性测试；更新版本一般亦可正常运行，但未做长期验证。

---

## 安装与快速上手

### 方式一：为普通用户（推荐）

为不熟悉代码的用户提供两种即开即用的软件包。

#### 选项 A：单文件版 (Standalone)
从 **[Google Drive 备份]([https://drive.google.com/file/d/1BvLx0z0h46n3n_obOIHizyThG2UIJ_rh/view?usp=drive_link](https://drive.google.com/file/d/10EBF0krNCyr6tQDBzn_Psdj1dyasO8Y-/view?usp=drive_link))** 下载 `SpectroEase.exe`。这是一个独立文件，无需安装，可直接运行。

#### 选项 B：文件夹版 (Folder)
从 **[GitHub Release]([https://github.com/shudayi/SpectroEase-V1.0/releases/tag/v1.0.0](https://github.com/shudayi/SpectroEase-V1.0))** 下载 `EXE for SpectroEase`。解压后运行其中的 `SpectroEase.exe`。此版本启动速度可能更快。

### 方式二：为开发者（从源码运行）

#### 1 · 克隆源码并安装依赖

```powershell
git clone https://github.com/shudayi/SpectroEase-V1.0
cd SpectroEase-V1.0
pip install -r requirements.txt   # 已固定精确版本
```

#### 2 · 启动应用

```powershell
python main.py                    # 打开 GUI
```

### 3 · 基础工作流

> **示意视频**  
> [点击在BiliBili 观看 »](https://www.bilibili.com/video/BV1tuWizxENX/?pop_share=1&vd_source=ee5073c103f10446477719780c85450d)
[![观看演示（SpectroEase 基础工作流）](https://www.bilibili.com/video/BV1tuWizxENX/?pop_share=1&vd_source=ee5073c103f10446477719780c85450d)

**推荐操作顺序**
1. **导入数据 / 划分**（Train/Test、K-Fold、Stratified K-Fold、LOGO 等）
2. **预处理配置**（基线校正、平滑、散射校正、归一化、导数、峰对齐、异常值检测）
3. **特征选择**（PCA / PLSR / 小波 / 寻峰）
4. **建模 / 超参优化**（SVM、RF、KNN、XGBoost、LightGBM、PLSR、SVR、GPR；Grid / Random / GA）
5. **评估与可视化**（ROC、混淆矩阵、特征重要性、回归残差等）
6. **导出报告**（PDF / Excel）




## 数据格式说明

| 维度        | 说明                                          |
| --------- | ------------------------------------------- |
| **支持格式**  | CSV · TXT · Excel                           |
| **推荐布局**  | 行式：`Sample_ID, Label, 400nm, 402nm, …`      |
| **标签关键字** | category · class · label · variety · target |
| **示例数据集** | `examples/` (CC-BY-4.0)           |


![image](https://github.com/user-attachments/assets/de2a0556-5729-48fc-b10f-c6efb85da488)

---

## 性能优化

* 多线程UI，确保界面响应流畅
* CPU多进程并行，加速密集计算
* 智能缓存，自动复用相同中间结果

---

## 许可证与免责声明

* **代码**：MIT License（见 `LICENSE.txt`）
* **示例数据**：CC-BY-4.0
* **免责声明**：本软件仅供科研与教学使用。作者不对商业用途或模型误判造成的任何损失负责；使用者需遵守当地法律法规。

---

**SpectroEase —— 让光谱分析变简单**
