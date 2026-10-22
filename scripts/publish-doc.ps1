#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 参数处理（安全方式）
if ($args.Count -gt 0) {
    $msg = $args[0]
}
else {
    $msg = "docs: update"
}

if ($args.Count -gt 1) {
    $branch = $args[1]
}
else {
    $branch = "doc"
}

# 检查git命令
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git not found." -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Pulling remote $branch ..." -ForegroundColor Cyan
git pull --rebase origin $branch

Write-Host "[1.5/5] Converting Jupyter notebooks ..." -ForegroundColor Cyan
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 scripts/convert-notebooks.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some notebooks conversion failed, continuing with existing content." -ForegroundColor Yellow
    }
}
else {
    Write-Host "Warning: py not found, skipping notebook conversion." -ForegroundColor Yellow
}

Write-Host "[2/5] Generating docs/index.json ..." -ForegroundColor Cyan
if (Get-Command node -ErrorAction SilentlyContinue) {
    node scripts/build-docs-index.js
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 scripts/build-docs-index.py
}
else {
    Write-Host "Error: Neither node nor py found." -ForegroundColor Red
    exit 1
}

Write-Host "[3/5] Staging changes ..." -ForegroundColor Cyan
git add -A

Write-Host "[4/5] Checking for changes to commit ..." -ForegroundColor Cyan
git diff --cached --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "No changes, skipping commit." -ForegroundColor Yellow
    exit 0
}

Write-Host "[5/5] Committing and pushing to $branch ..." -ForegroundColor Cyan
git commit -m $msg
git push origin $branch

Write-Host "Done: Pushed to $branch" -ForegroundColor Green