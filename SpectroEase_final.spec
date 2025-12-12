# -*- mode: python ; coding: utf-8 -*-
# SpectroEase 最终打包配置 - 只打包requirements.txt中的依赖
# 避免打包不必要的大型库（torch, tensorflow, cv2等）

import os
import sys

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('logo.png', '.'),
        ('dark_theme.qss', '.'),  # 主题文件
        ('config', 'config'),
        ('translations', 'translations'),
        ('examples', 'examples'),
        ('interfaces', 'interfaces'),
        ('plugins', 'plugins')
    ],
    hiddenimports=[
        # 插件模块
        'plugins.data_partitioning.advanced_splitter',
        'plugins.data_partitioning.stratified_splitter',
        'plugins.feature_selection.advanced_feature_selector',
        'plugins.feature_selection.pca',
        'plugins.feature_selection.spectrum_feature_extractor',
        'plugins.feature_selection.unsupervised_pca',
        'plugins.modeling.advanced_modeling',
        'plugins.modeling.qualitative_analyzer',
        'plugins.modeling.quantitative_analyzer',
        'plugins.modeling.sklearn_debug_tracer',
        'plugins.preprocessing.custom_preprocessing',
        'plugins.preprocessing.snv_processor',
        'plugins.preprocessing.spectrum_converter',
        'plugins.preprocessing.spectrum_preprocessor',
        'plugins.preprocessing.spectrum_visualizer',
        'plugins.preprocessing.standard_scaler',
        'plugins.reporting.spectrum_report_generator',
        # scikit-learn
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree',
        'sklearn.tree._utils',
        'sklearn.utils._weight_vector',
        # scipy
        'scipy._lib.messagestream',
        'scipy.special._ufuncs_cxx',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.special.cython_special',
        # pyqtgraph
        'pyqtgraph.Qt',
        'pyqtgraph.graphicsItems',
        # pandas
        'pandas._libs.tslibs.base',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        # 其他
        'joblib',
        'lightgbm',
        'xgboost',
        'onnx',
        'onnxruntime',
        'skl2onnx',
        'deap',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不在requirements.txt中的大型库
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'tf',
        'cv2',
        'transformers',
        'timm',
        'datasets',
        'skimage',
        'pydantic',
        'sentry_sdk',
        'opentelemetry',
        'shapely',
        'plotly',
        'dash',
        # 排除不需要的标准库模块
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'test',
        # 注意：不能排除 'unittest'，因为 numpy.testing 需要它
        # 'unittest',  # 保留，numpy/scipy 需要
        'distutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 确保 Python DLL 被正确包含
# PyInstaller 通常会自动包含 Python DLL，但这里显式检查以确保
python_dlls = [x for x in a.binaries if 'python' in x[0].lower() and x[0].endswith('.dll')]
print(f"[DEBUG] 找到的 Python DLL: {[x[0] for x in python_dlls]}")

# 如果未找到 Python DLL，尝试从 Python 安装目录添加
if not python_dlls:
    python_dll_path = None
    # 检查 Python 安装目录
    python_dir = os.path.dirname(sys.executable)
    potential_dlls = [
        os.path.join(python_dir, 'python311.dll'),
        os.path.join(python_dir, 'python3.dll'),
        os.path.join(sys.prefix, 'python311.dll'),
        os.path.join(sys.prefix, 'python3.dll'),
    ]
    
    for dll_path in potential_dlls:
        if os.path.exists(dll_path):
            python_dll_path = dll_path
            print(f"[DEBUG] 找到 Python DLL: {python_dll_path}")
            # 检查是否已经在 binaries 列表中
            dll_name = os.path.basename(dll_path)
            if not any(dll_name.lower() == x[0].lower() for x in a.binaries):
                a.binaries.append((dll_name, dll_path, 'BINARY'))
                print(f"[DEBUG] 已添加 Python DLL: {dll_name}")
            break
    
    if not python_dll_path:
        print("[WARNING] 未找到 Python DLL，PyInstaller 应该会自动包含")

# Python 字节码打包（压缩）
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 创建 EXE（不包含依赖库）- onedir模式
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 关键：排除二进制文件，使其独立存储
    name='SpectroEase',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # 禁用UPX压缩
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.png',
)

# 收集所有文件到目录（onedir模式）
coll = COLLECT(
    exe,
    a.binaries,      # 所有二进制依赖（DLL等）
    a.zipfiles,      # ZIP文件
    a.datas,         # 数据文件
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpectroEase',
)



