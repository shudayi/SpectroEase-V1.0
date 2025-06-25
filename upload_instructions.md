# 使用 GitHub Token 上传项目

## 🔑 步骤一：获取 GitHub Personal Access Token

1. 登录 GitHub → 点击右上角头像 → Settings
2. 左侧菜单：Developer settings → Personal access tokens → Tokens (classic)
3. 点击 "Generate new token (classic)"
4. 填写信息：
   - Note: `SpectroEase Upload`
   - Expiration: 选择适当的过期时间
   - 权限：勾选 `repo` (Full control of private repositories)
5. 点击 "Generate token"
6. **重要：立即复制 token，页面刷新后将无法再次查看**

## 🚀 步骤二：运行上传脚本

将您的 token 替换到下面的命令中：

```powershell
.\github_upload.ps1 -Token "YOUR_GITHUB_TOKEN_HERE"
```

例如：
```powershell
.\github_upload.ps1 -Token "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
```

## 📋 脚本功能说明

该脚本会自动：
- ✅ 验证仓库访问权限
- 📁 扫描项目文件（自动排除临时文件）
- 📤 批量上传到 GitHub
- 📊 显示上传进度和结果

## 🔍 自动排除的文件类型

- `.git*` 文件
- `*.ps1` 脚本文件
- `__pycache__` 和 `*.pyc` Python缓存
- `debug_data` 调试数据
- `reports` 报告文件
- `*.zip` 压缩文件
- 大于 100MB 的文件

## ⚡ 优势

- 🚫 无需安装 Git
- 🔒 安全的 token 认证
- 📊 实时上传进度显示
- 🛡️ 自动错误处理
- ⏱️ 防止 API 速率限制

## 🆘 如果遇到问题

1. **Token 权限错误**：确保 token 有 `repo` 权限
2. **仓库不存在**：先在 GitHub 创建空仓库
3. **文件过大**：大文件会被自动跳过
4. **网络问题**：脚本会显示详细错误信息

---

**准备好了吗？请提供您的 GitHub Token，我来帮您上传！** 