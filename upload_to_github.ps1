# PowerShell脚本上传SpectroEase到GitHub
Write-Host "开始上传SpectroEase到GitHub..." -ForegroundColor Green

# 设置Git别名
Set-Alias -Name git -Value "C:\Program Files\Git\bin\git.exe" -Scope Global

try {
    # 禁用分页器
    & git config core.pager ""
    
    # 初始化Git LFS
    Write-Host "配置Git LFS..." -ForegroundColor Yellow
    & git lfs install
    & git lfs track "*.exe"
    
    # 检查状态
    Write-Host "检查仓库状态..." -ForegroundColor Yellow
    $status = & git status --porcelain
    
    if ($status) {
        # 分批添加文件
        Write-Host "添加配置文件..." -ForegroundColor Yellow
        & git add .gitignore .gitattributes
        
        Write-Host "添加主要代码..." -ForegroundColor Yellow
        & git add app/ config/ interfaces/ plugins/ utils/ translations/
        
        Write-Host "添加示例和文档..." -ForegroundColor Yellow
        & git add examples/ main.py requirements.txt README.md logo.png
        
        Write-Host "添加大文件..." -ForegroundColor Yellow
        & git add *.exe SpectroEase_Release/
        
        # 提交
        Write-Host "提交更改..." -ForegroundColor Yellow
        & git commit -m "Upload complete SpectroEase project with LFS support"
        
        # 推送
        Write-Host "推送到GitHub..." -ForegroundColor Yellow
        & git push origin main
        
        Write-Host "上传完成！" -ForegroundColor Green
    } else {
        Write-Host "没有需要上传的更改" -ForegroundColor Yellow
    }
} catch {
    Write-Host "错误: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "按回车键退出"
