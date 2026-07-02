#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

<#
.SYNOPSIS
  One-click environment setup for all Python projects in this repo.
  Creates .venv, installs dependencies, and copies .env.example if .env missing.

.DESCRIPTION
  Scans known projects, creates a virtual environment in each, installs
  their requirements, and bootstraps .env from .env.example when needed.

  Usage:
    .\scripts\setup-env.ps1          # set up all projects
    .\scripts\setup-env.ps1 -Project book_search  # single project
#>

param(
    [ValidateSet("book_search", "svd_quant", "ContinuePretrain", "all")]
    [string]$Project = "all"
)

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }

# ---- Project definitions ----
$projects = @(
    @{
        Name     = "book_search"
        Path     = "project/book_search"
        Reqs     = "requirements.txt"
        ReqsDev  = "requirements-dev.txt"
        EnvFiles = @("src/llm/.env")
        EnvEx    = "src/llm/.env.example"
    }
    @{
        Name     = "svd_quant"
        Path     = "project/quant/svd_quant"
        Reqs     = $null   # no requirements.txt; manual pip install torch numpy
        ReqsDev  = $null
        EnvFiles = @()
        EnvEx    = $null
    }
    @{
        Name     = "ContinuePretrain"
        Path     = "project/ContinuePretrain"
        Reqs     = $null
        ReqsDev  = $null
        EnvFiles = @()
        EnvEx    = $null
    }
)

# ---- Filter ----
$targets = if ($Project -eq "all") { $projects } else { $projects | Where-Object { $_["Name"] -eq $Project } }
if (-not $targets) { Write-Host "Unknown project: $Project"; exit 1 }

$ok = $true
foreach ($p in $targets) {
    Write-Host "`n===== $($p.Name) =====" -ForegroundColor Cyan
    $projDir = Join-Path $RepoRoot $p.Path

    if (-not (Test-Path $projDir)) {
        Write-Host "  SKIP: directory not found ($($p.Path))" -ForegroundColor Yellow
        continue
    }

    # ---- Virtual environment ----
    $venvDir = Join-Path $projDir ".venv"
    if (-not (Test-Path $venvDir)) {
        Write-Host "  Creating .venv ..." -ForegroundColor Gray
        & cmd /c "$py -m venv `"$venvDir`""
        if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED to create venv" -ForegroundColor Red; $ok = $false; continue }
    } else {
        Write-Host "  .venv already exists, skip" -ForegroundColor Gray
    }

    # ---- Activate helper ----
    $pip = Join-Path $venvDir "Scripts\pip.exe"

    # ---- Install requirements ----
    if ($p.Reqs) {
        $reqFile = Join-Path $projDir $p.Reqs
        if (Test-Path $reqFile) {
            Write-Host "  Installing $($p.Reqs) ..." -ForegroundColor Gray
            & $pip install -r $reqFile
            if ($LASTEXITCODE -ne 0) { Write-Host "  WARN: pip install had issues" -ForegroundColor Yellow }
        }
    }
    if ($p.ReqsDev) {
        $reqDevFile = Join-Path $projDir $p.ReqsDev
        if (Test-Path $reqDevFile) {
            Write-Host "  Installing $($p.ReqsDev) ..." -ForegroundColor Gray
            & $pip install -r $reqDevFile
        }
    }

    # ---- Bootstrap .env from example ----
    foreach ($envRel in $p.EnvFiles) {
        $envFile = Join-Path $projDir $envRel
        $exFile  = Join-Path $projDir $p.EnvEx
        if (( -not (Test-Path $envFile)) -and (Test-Path $exFile)) {
            Copy-Item -LiteralPath $exFile -Destination $envFile
            Write-Host "  Created $envRel from .env.example" -ForegroundColor Green
        }
    }
}

if ($ok) {
    Write-Host "`nDone. Activate a venv with:" -ForegroundColor Green
    foreach ($p in $targets) {
        Write-Host "  .\$($p.Path)\.venv\Scripts\Activate.ps1"
    }
} else {
    exit 1
}
