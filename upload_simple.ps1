# 简化的上传脚本
Write-Host "Starting upload to GitHub..." -ForegroundColor Green

# 设置Git别名
$env:PATH += ";C:\Program Files\Git\bin"

# 禁用分页器
git config core.pager ""

# 初始化Git LFS
Write-Host "Setting up Git LFS..." -ForegroundColor Yellow
git lfs install
git lfs track "*.exe"

# 添加文件
Write-Host "Adding files..." -ForegroundColor Yellow
git add .

# 提交
Write-Host "Committing..." -ForegroundColor Yellow
git commit -m "Upload complete SpectroEase project with LFS support"

# 推送
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "Upload completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
