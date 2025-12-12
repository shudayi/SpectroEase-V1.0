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

## 🆕 V1.2.1 New Features (Latest Update)

### Professional Spectral-Specific Preprocessing ⭐
- **Raman专用**: Fluorescence background removal (ModPoly, VRA, AFBS) · Raman Shift calibration
- **MIR/FTIR专用**: Atmospheric compensation (CO₂ and H₂O interference removal)
- **NIR专用**: Water peak removal (EPO, DOSC algorithms)  
- **Enhanced Baseline**: ModPoly and SNIP baseline correction algorithms
- **Model Transfer**: PDS and SBC for inter-instrument calibration
- **New UI Organization**: Preprocessing tab organized by spectral type (Universal / Raman / MIR / NIR / Advanced)

### Intelligent Display Adaptation 🎨
- **Auto DPI Detection**: Automatically adapts to 1080p, 2K, 4K, and Retina displays (150-200 DPI)
- **Dynamic Font Scaling**: UI fonts adjust based on screen resolution for optimal readability
- **Multi-Monitor Support**: Seamless experience across different display configurations
- **Zero Configuration**: Works perfectly out-of-the-box on all modern displays

---

## Key Features

**​​Multi-format Data Import**​​​​: CSV (optimal performance) / TXT / Excel formats with automated label recognition and data validation
**​​​​Advanced Preprocessing**​​​​: Baseline correction, smoothing, scatter correction, normalization, derivative transformation, peak alignment, and spectral-specific algorithms (Raman/MIR/NIR)
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
├── main.py                 # Application entry point
├── app/                    # Main application
│   ├── controllers/        # Controller layer
│   ├── models/            # Data models
│   ├── services/          # Business logic services
│   ├── views/             # User interface
│   └── utils/             # Utility functions
├── plugins/               # Plugin modules
│   ├── preprocessing/     # Preprocessing algorithms
│   ├── feature_selection/ # Feature selection methods
│   ├── modeling/          # Machine learning models
│   └── reporting/         # Report generation
├── interfaces/            # Interface definitions
├── config/                # Configuration files
├── examples/              # Example datasets
└── translations/          # Multi-language support
```

---

## Algorithm Catalogue

### 1 · Data Partitioning

* Train–Test Split
* K-Fold Cross-Validation
* Stratified K-Fold
* Leave-One-Group-Out (LOGO)
* Random Split

### 2 · Pre-processing

| Category                 | Algorithms                                                                   |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Despiking**            | MAD (Median Absolute Deviation) · Local Z-score                             |
| **Baseline Correction**  | Polynomial · ALS · airPLS · **ModPoly ⭐** · **SNIP ⭐**                   |
| **Scatter Correction**   | SNV · MSC · EMSC · RNV · OSC                                                 |
| **Smoothing**            | Savitzky–Golay · Moving Average · Median Filter · Gaussian · Wavelet        |
| **Raman-Specific ⭐**    | **Fluorescence Removal (ModPoly/VRA/AFBS)** · **Raman Shift Calibration**  |
| **MIR-Specific ⭐**      | **Atmospheric Compensation (CO₂/H₂O Removal)**                             |
| **NIR-Specific ⭐**      | **Water Peak Removal (EPO/DOSC)**                                           |
| **Model Transfer ⭐**    | **PDS (Piecewise Direct Standardization)** · **SBC (Slope-Bias Correction)**|
| **Scaling & Enhancement**| Standard Scale · Min–Max Scale · L2 Normalize · Vector · Area · Maximum · First Derivative · Second Derivative · Savitzky–Golay Derivative · Finite Difference · Gap-Segment · Denoising · Peak Alignment · Outlier Detection |

### 3 · Feature Selection

| Category                 | Algorithms                                                                   |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Wavelength Selection ⭐**| **CARS (Competitive Adaptive Reweighted Sampling)** · **SPA (Successive Projections Algorithm)** |
| **Statistical Methods**  | SelectKBest · Mutual Information · Information Gain · Correlation Filter · Variance Threshold |
| **Model-based Methods**  | RFE · Feature Importance · LASSO · PLS Regression                           |
| **Transform Methods**    | PCA · Wavelet Transform                                                      |
| **Optimization Methods** | Genetic Algorithm · Spectral Optimized                                      |
| **Spectroscopy-specific**| Peak Detection                                                               |

### 4 · Modelling

#### 4.1 Classification

Logistic Regression · SVM · KNN · Decision Tree · Random Forest · Extra Trees · Gradient Boosting · AdaBoost · Neural Network (MLP) · Naive Bayes

#### 4.2 Regression

Linear Regression · Ridge · Lasso · ElasticNet · SVR · KNN Regressor · Decision Tree Regressor · Random Forest Regressor · Extra Trees Regressor · Gradient Boosting Regressor · AdaBoost Regressor · Neural Network (MLP)

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
| PyQt5        | 5.15.7         | Graphical user interface       |
| NumPy        | 2.3.2          | Numerical computing            |
| pandas       | 2.3.1          | Data manipulation              |
| SciPy        | 1.16.1         | Scientific algorithms          |
| scikit-learn | 1.7.1          | Machine-learning workflows     |
| Matplotlib   | 3.10.5         | Visualisation                  |
| seaborn      | 0.13.2         | Statistical visualization      |
| pyqtgraph    | 0.13.7         | Interactive plotting           |
| ReportLab    | 4.4.3          | PDF / report generation        |
| OpenPyXL     | 3.1.5          | Excel file handling            |
| xlrd         | 2.0.2          | Excel file reading             |
| deap         | 1.4.3          | Genetic algorithms             |
| requests     | 2.32.5         | HTTP library                   |
| Pillow       | 11.3.0         | Image processing               |
| joblib       | 1.5.1          | Parallel computing             |
| threadpoolctl| 3.6.0          | Thread pool control            |

**Note**: The table above lists the main dependencies. Additional supporting libraries (contourpy, cycler, fonttools, kiwisolver, pyparsing, urllib3, certifi, idna, charset-normalizer, et-xmlfile, packaging, setuptools, six, python-dateutil, pytz, tzdata) are automatically installed as sub-dependencies. All versions satisfy SpectroEase's compatibility requirements; newer versions should work but are not actively tested.

---

## Installation & Quick Start

**⚠️ If the exe file download fails, please try these alternative links:**
- **[GitHub Release](https://github.com/shudayi/SpectroEase-V1.0/releases/tag/v1.0.0)**
- **[Google Drive Backup](https://drive.google.com/file/d/1AMFtOIimYQIzwiuXrtrno-7OxNDhqzXh/view?usp=drive_link)**

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
├── main.py                 # 应用程序入口
├── app/                    # 主应用程序
│   ├── controllers/        # 控制器层
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑服务
│   ├── views/             # 用户界面
│   └── utils/             # 工具函数
├── plugins/               # 插件模块
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
| **统计方法**    | SelectKBest · Mutual Information · Information Gain · Correlation Filter · Variance Threshold |
| **模型方法**    | RFE · Feature Importance · LASSO · PLS Regression                            |
| **变换方法**    | PCA · Wavelet Transform                                                       |
| **优化方法**    | Genetic Algorithm · Spectral Optimized                                       |
| **光谱专用**    | Peak Detection                                                                |

### 4 · 建模

#### 4.1 定性分析（分类）

Logistic Regression · SVM · KNN · Decision Tree · Random Forest · Extra Trees · Gradient Boosting · AdaBoost · Neural Network (MLP) · Naive Bayes

#### 4.2 定量分析（回归）

Linear Regression · Ridge · Lasso · ElasticNet · SVR · KNN Regressor · Decision Tree Regressor · Random Forest Regressor · Extra Trees Regressor · Gradient Boosting Regressor · AdaBoost Regressor · Neural Network (MLP)

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
| PyQt5        | 5.15.7   | 图形界面                   |
| NumPy        | 2.3.2    | 数值计算                   |
| pandas       | 2.3.1    | 数据处理                   |
| SciPy        | 1.16.1   | 科学计算                   |
| scikit-learn | 1.7.1    | 机器学习框架                 |
| Matplotlib   | 3.10.5   | 可视化                    |
| seaborn      | 0.13.2   | 统计可视化                  |
| pyqtgraph    | 0.13.7   | 交互式绘图                  |
| ReportLab    | 4.4.3    | PDF / 报告生成             |
| OpenPyXL     | 3.1.5    | Excel 文件处理             |
| xlrd         | 2.0.2    | Excel 文件读取             |
| deap         | 1.4.3    | 遗传算法                   |
| requests     | 2.32.5   | HTTP库                  |
| Pillow       | 11.3.0   | 图像处理                   |
| joblib       | 1.5.1    | 并行计算                   |
| threadpoolctl| 3.6.0    | 线程池控制                  |

**说明**：上表列出了主要依赖库。其他支持库（contourpy, cycler, fonttools, kiwisolver, pyparsing, urllib3, certifi, idna, charset-normalizer, et-xmlfile, packaging, setuptools, six, python-dateutil, pytz, tzdata）作为子依赖会自动安装。上述版本均通过兼容性测试；更新版本一般亦可正常运行，但未做长期验证。

---

## 安装与快速上手

**⚠️ 如果exe文件下载失败，请尝试使用这些备用链接：**
- **[GitHub Release](https://github.com/shudayi/SpectroEase-V1.0/releases/tag/v1.0.0)**
- **[Google Drive 备份](https://drive.google.com/file/d/1AMFtOIimYQIzwiuXrtrno-7OxNDhqzXh/view?usp=drive_link)**

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

