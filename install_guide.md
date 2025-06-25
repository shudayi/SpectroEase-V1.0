# Git 安装指南

## 方法一：手动下载安装（推荐）

1. 访问 Git 官方下载页面：
   https://git-scm.com/download/win

2. 点击下载 Git for Windows（会自动下载最新版本）

3. 运行下载的安装程序（例如：`Git-2.43.0-64-bit.exe`）

4. 安装过程中的重要设置：
   - 选择安装路径（默认即可）
   - 选择组件：勾选 "Git Bash Here" 和 "Git GUI Here"
   - 选择默认编辑器：可以选择 Notepad++ 或 VS Code
   - 调整PATH环境：选择 "Git from the command line and also from 3rd-party software"
   - 选择HTTPS传输后端：使用 OpenSSL 库
   - 配置行结束符：选择 "Checkout Windows-style, commit Unix-style line endings"
   - 选择终端模拟器：使用 MinTTY
   - 其他选项保持默认

5. 完成安装后，重新启动 PowerShell 或命令提示符

## 方法二：使用浏览器直接下载

如果上述链接无法访问，可以直接下载：
https://github.com/git-for-windows/git/releases/latest

选择以 `.exe` 结尾的安装程序文件。

## 验证安装

安装完成后，在新的 PowerShell 窗口中运行：
```powershell
git --version
```

如果显示版本信息，说明安装成功。

## 下一步

安装完成后，运行我们创建的脚本：
```powershell
.\setup_git.ps1
``` 