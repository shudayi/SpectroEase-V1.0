# SpectroEase — A Visual Workflow for Spectral Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **SpectroEase** is an open-source, extensible desktop application.
> It converts the entire pipeline—**data loading / splitting → preprocessing → feature selection → modelling (with hyper-parameter optimisation) → evaluation → reporting**—into a drag-and-drop visual workflow, enabling scientists and engineers to perform qualitative or quantitative spectral analysis with zero coding.

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
10. [Citation](#citation)
11. [Licence & Disclaimer](#licence--disclaimer)

</details>

---

## Key Features

* **Multi-format import** — CSV / TXT / Excel with automatic label detection & validation
* **Advanced preprocessing** — baseline correction, smoothing, scatter correction, normalisation, derivatives, peak alignment
* **Feature selection** — SelectKBest, RFE, LASSO, PCA, PLSR, wavelets, automatic peak detection
* **Modelling** — 30 + built-in classifiers, regressors and clustering models
* **Hyper-parameter optimisation** — grid search / random search / genetic algorithm
* **Visual evaluation** — ROC curves, confusion matrices, feature-importance plots, residual plots
* **Plugin architecture** — core steps are modular and easy to extend
* **Guided UI** — wizard-style workflow plus parameter panels for a gentle learning curve

---

## UI Overview


|             Main Window             | 
| :---------------------------------: | 
|![image](https://github.com/user-attachments/assets/b8639d31-b265-4200-8c27-c935ab65daed)
|            Pipeline                 | 
| ![image](https://github.com/user-attachments/assets/0e819ef5-3819-43ba-987c-0b36abe8f739)

---

## System Architecture

SpectroEase adopts a **modular plugin architecture** with clear separation of concerns:

```text
SpectroEase/
├── app/                    # Main application
│   ├── main.py             # Application entry point
│   ├── gui/                # User interface components
│   ├── core/               # Core business logic
│   └── utils/              # Utility functions
├── plugins/                # Plugin modules
│   ├── preprocessing/      # Pre-processing plugins
│   ├── feature_selection/  # Feature-selection plugins
│   ├── modeling/           # Machine-learning plugins
│   └── data_partition/     # Data-partitioning plugins
├── interfaces/             # Plugin interfaces
│   ├── base_plugin.py
│   ├── preprocessing_interface.py
│   ├── feature_selection_interface.py
│   ├── modeling_interface.py
│   └── data_partition_interface.py
└── config/                 # Configuration files
```

This layout lets you drop new plugins into the relevant folder with minimal boilerplate—every plugin implements one of the interfaces in `interfaces/`, ensuring compatibility with the main GUI.

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

* SelectKBest *(f\_classif · mutual\_info\_classif · χ²)*
* Recursive Feature Elimination (RFE)
* LASSO · Mutual Information · Feature Importance
* PCA · PLSR · Automatic Peak Detection · Wavelet Transform

### 4 · Modelling

#### 4.1 Qualitative (Classification / Clustering)

Logistic Regression · LDA / QDA · SVM · KNN · Decision Tree · Random Forest · Gradient Boosting · XGBoost · Neural Network · K-Means · Hierarchical · DBSCAN

#### 4.2 Quantitative (Regression)

MLR · PLSR · SVR · Decision Tree Regression · Random Forest Regression · GPR · Ridge · Lasso · ElasticNet

### 5 · Hyper-parameter Optimisation

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
# or run a batch-workflow example
python main.py --workflow examples/pca_svm.yml
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

> **SpectroEase** 是一款开源、可扩展的桌面应用程序。
> 它将 **数据载入/分割 → 预处理 → 特征选择 → 建模（含超参数优化） → 评估 → 报告** 封装为拖拽式工作流，帮助科研人员和工程师零代码完成光谱定性/定量分析。

---

<details>
<summary><strong>目录（点击展开）</strong></summary>

1. [关键特性](#关键特性)
2. [界面概览](#界面概览)
3. [系统架构](#系统架构)
4. [完整算法清单](#完整算法清单)
5. [系统需求](#系统需求)
6. [依赖库](#依赖库)
7. [安装与快速上手](#安装与快速上手)
8. [数据格式说明](#数据格式说明)
9. [性能优化](#性能优化)
10. [引用方式](#引用方式)
11. [许可证与免责声明](#许可证与免责声明)

</details>

---

## 关键特性

* **多格式导入**：CSV / TXT / Excel，自动标签识别与数据校验
* **高级预处理**：基线校正、平滑、散射校正、归一化、导数、波峰对齐
* **特征选择**：SelectKBest、RFE、LASSO、PCA、PLSR、小波、自动峰检
* **建模算法**：定性、定量两大类 30 + 模型内置支持
* **超参数优化**：网格搜索 / 随机搜索 / 遗传算法
* **可视化评估**：ROC、混淆矩阵、特征重要性、回归残差等丰富图表
* **插件架构**：核心功能插件化，便于二次开发
* **界面模式**：向导式工作流 + 参数面板，降低上手门槛

---

## 界面概览



|               主界面               |                   预处理流程                   |
| :-----------------------------: | :---------------------------------------: |
| 如英文区 |

---

## 系统架构

SpectroEase 采用 **模块化插件架构**，各功能职责分离：

```text
SpectroEase/
├── app/                    # 主程序
│   ├── main.py             # 入口脚本
│   ├── gui/                # UI 组件
│   ├── core/               # 核心业务逻辑
│   └── utils/              # 工具函数
├── plugins/                # 插件
│   ├── preprocessing/      # 预处理插件
│   ├── feature_selection/  # 特征选择插件
│   ├── modeling/           # 机器学习插件
│   └── data_partition/     # 数据分割插件
├── interfaces/             # 插件接口定义
│   ├── base_plugin.py
│   ├── preprocessing_interface.py
│   ├── feature_selection_interface.py
│   ├── modeling_interface.py
│   └── data_partition_interface.py
└── config/                 # 配置文件
```

通过继承接口并放置于对应目录，即可快捷开发并加载新插件。

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

* SelectKBest *(f\_classif · mutual\_info\_classif · χ²)*
* Recursive Feature Elimination (RFE)
* LASSO · Mutual Information · Feature Importance
* PCA · PLSR · 自动峰检 · Wavelet Transform

### 4 · 建模

#### 4.1 定性分析

Logistic Regression · LDA / QDA · SVM · KNN · Decision Tree · Random Forest · Gradient Boosting · XGBoost · Neural Network · K-Means · Hierarchical · DBSCAN

#### 4.2 定量分析

MLR · PLSR · SVR · Decision Tree Regression · Random Forest Regression · GPR · Ridge · Lasso · ElasticNet

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

---

## 数据格式说明

| 维度        | 说明                                          |
| --------- | ------------------------------------------- |
| **支持格式**  | CSV · TXT · Excel                           |
| **推荐布局**  | 行式：`Sample_ID, Label, 400nm, 402nm, …`      |
| **标签关键字** | category · class · label · variety · target |
| **示例数据集** | `datasets/seed_demo/` (CC-BY-4.0)           |

---






---

## 许可证与免责声明

* **代码**：MIT License（见 `LICENSE.txt`）
* **示例数据**：CC-BY-4.0
* **免责声明**：本软件仅供科研与教学使用。作者不对商业用途或模型误判造成的任何损失负责；使用者需遵守当地法律法规。

---

**SpectroEase —— 让光谱分析变简单**
