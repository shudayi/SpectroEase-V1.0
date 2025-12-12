# PowerShell script to clean up the project before building

# Remove all __pycache__ directories
Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# Remove all .pyc files
Get-ChildItem -Path . -Include *.pyc -Recurse | Remove-Item -Force

# Clear the logs directory
Get-ChildItem -Path "logs" -Recurse | Remove-Item -Recurse -Force

# Clear the debug_data directory
Get-ChildItem -Path "debug_data" -Recurse | Remove-Item -Recurse -Force

# Remove build and dist directories
Remove-Item -Recurse -Force -Path "build", "dist"

Write-Host "Cleanup complete."