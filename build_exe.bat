@echo off
REM SpectroEase EXE打包脚本
REM 使用方法：双击运行或在命令行执行 build_exe.bat

echo ========================================
echo   SpectroEase 打包工具
echo ========================================
echo.

REM 检查Python环境（优先使用py launcher，兼容python命令）
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [错误] 未找到Python环境！
        echo 请确保已安装Python并添加到PATH环境变量
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
    set PIP_CMD=pip
) else (
    set PYTHON_CMD=py
    set PIP_CMD=py -m pip
)

echo [1/5] 检查依赖包...
%PIP_CMD% show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装PyInstaller...
    %PIP_CMD% install pyinstaller
)

echo.
echo [2/5] 清理旧的构建文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "__pycache__" rmdir /s /q "__pycache__"
echo 清理完成

echo.
echo [3/5] 开始打包（这可能需要几分钟）...
echo 使用配置文件: SpectroEase_final.spec
echo.

%PYTHON_CMD% -m PyInstaller --clean --noconfirm SpectroEase_final.spec

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    echo 请检查上面的错误信息
    pause
    exit /b 1
)

echo.
echo [4/5] 检查打包结果...
if exist "dist\SpectroEase\SpectroEase.exe" (
    echo [成功] 找到可执行文件
) else (
    echo [错误] 未找到可执行文件
    pause
    exit /b 1
)

echo.
echo [5/5] 复制必要文件...
REM 复制README等文档
if exist "README.md" copy "README.md" "dist\SpectroEase\" >nul
if exist "LICENSE" copy "LICENSE" "dist\SpectroEase\" >nul

echo.
echo ========================================
echo   打包完成！
echo ========================================
echo.
echo 可执行文件位置: dist\SpectroEase\SpectroEase.exe
echo.
echo 您可以：
echo 1. 直接运行 dist\SpectroEase\SpectroEase.exe
echo 2. 将整个 dist\SpectroEase 文件夹复制到其他电脑
echo 3. 使用打包工具创建安装程序
echo.
pause



