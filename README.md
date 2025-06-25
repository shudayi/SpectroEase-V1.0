# SpectroEase V1.0

一个专业的光谱数据分析和机器学习平台，为光谱学研究提供全面的数据处理、特征选择、建模和评估工具。

## 主要功能

### 🔬 数据处理
- 支持多种光谱数据格式导入
- 智能数据预处理和标准化
- 数据可视化和探索性分析

### 🛠️ 预处理
- SNV (Standard Normal Variate) 标准化
- MSC (Multiplicative Scatter Correction) 散射校正
- SG (Savitzky-Golay) 平滑滤波
- 导数计算和基线校正

### 🎯 特征选择
- PCA 主成分分析
- 变量重要性分析
- 光谱特征提取
- 高级特征选择算法

### 🤖 机器学习建模
- 定性分析模型
- 定量分析模型
- 高级建模算法集成
- 超参数优化

### 📊 模型评估
- 交叉验证
- 性能指标计算
- 可视化评估报告
- 模型比较分析

### 🔧 高级功能
- 插件系统支持
- ONNX 模型导出
- 多语言支持
- 配置管理

## 系统要求

- Python 3.8+
- PyQt5/PySide2
- scikit-learn
- numpy, pandas
- matplotlib, seaborn

## 安装与使用

1. 克隆仓库:
```bash
git clone https://github.com/shudayi/SpectroEase-V1.0.git
cd SpectroEase-V1.0
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 运行程序:
```bash
python main.py
```

## 项目结构

```
SpectroEase/
├── app/                    # 主应用程序
│   ├── controllers/        # 控制器层
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑服务
│   ├── utils/             # 工具函数
│   └── views/             # 用户界面
├── config/                # 配置文件
├── interfaces/            # 接口定义
├── plugins/               # 插件模块
├── translations/          # 多语言支持
└── utils/                 # 通用工具
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。 