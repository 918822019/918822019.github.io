param(
    [string[]]$Models = @("Qwen3.5-0.8B", "Qwen3.5-9B", "Qwen3.5-35B-A3B"),
    [ValidateSet("modelscope", "huggingface", "auto")]
    [string]$Source = "auto",
    [switch]$Help
)

function Show-Help {
@"
 从 ModelScope / HuggingFace 同步模型权重到 data/models/

用法: .\scripts\data\sync_models.ps1 [选项]

选项:
  -Models  要同步的模型列表 (默认: Qwen3.5-0.8B, Qwen3.5-9B, Qwen3.5-35B-A3B)
  -Source  下载源: modelscope / huggingface / auto (默认 auto，优先 modelscope 失败则回退到 huggingface)

示例:
  .\scripts\data\sync_models.ps1
  .\scripts\data\sync_models.ps1 -Source modelscope
  .\scripts\data\sync_models.ps1 -Source huggingface -Models Qwen3.5-0.8B
"@
}

if ($Help) {
    Show-Help
    return
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path "$scriptDir/../.."
$modelsDir = Join-Path $rootDir "data/models"

$modelscopeRepoMap = @{
    "Qwen3.5-0.8B"    = "Qwen/Qwen3.5-0.8B"
    "Qwen3.5-9B"      = "Qwen/Qwen3.5-9B"
    "Qwen3.5-35B-A3B" = "Qwen/Qwen3.5-35B-A3B"
}

$huggingfaceRepoMap = @{
    "Qwen3.5-0.8B"    = "Qwen/Qwen3.5-0.8B"
    "Qwen3.5-9B"      = "Qwen/Qwen3.5-9B"
    "Qwen3.5-35B-A3B" = "Qwen/Qwen3.5-35B-A3B"
}

function Download-ModelFromModelScope {
    param([string]$RepoId, [string]$TargetDir)

    $parentDir = Split-Path $TargetDir -Parent
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null

    Write-Host "  通过 modelscope CLI 下载..."
    & modelscope download $RepoId --local-dir $TargetDir

    if ($LASTEXITCODE -eq 0) {
        return $true
    }

    Write-Host "  modelscope CLI 失败，尝试 Python API..."
    try {
        & py -3 -c @"
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('$RepoId', cache_dir='$parentDir/.ms_cache')
import shutil, os, glob
src = glob.glob(os.path.join('$parentDir/.ms_cache', '**', '*'), recursive=True)[0]
src = os.path.dirname(src) if os.path.isfile(src) else src
if os.path.isdir(src):
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), os.path.join('$TargetDir', f))
"@
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Download-ModelFromHuggingFace {
    param([string]$RepoId, [string]$TargetDir)

    $parentDir = Split-Path $TargetDir -Parent
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null

    Write-Host "  通过 huggingface-cli 下载..."
    & huggingface-cli download $RepoId --local-dir $TargetDir --local-dir-use-symlinks False

    return $LASTEXITCODE -eq 0
}

foreach ($model in $Models) {
    $targetDir = Join-Path $modelsDir $model
    $msRepo = $modelscopeRepoMap[$model]
    $hfRepo = $huggingfaceRepoMap[$model]

    if (-not $msRepo -and -not $hfRepo) {
        Write-Warning "未知模型: $model，跳过"
        continue
    }

    if (Test-Path (Join-Path $targetDir "config.json")) {
        Write-Host "[跳过] $model 已存在于 $targetDir"
        continue
    }

    $success = $false

    if ($Source -eq "modelscope" -or $Source -eq "auto") {
        if ($msRepo) {
            Write-Host "[下载] $model 从 ModelScope ($msRepo)..."
            $success = Download-ModelFromModelScope -RepoId $msRepo -TargetDir $targetDir
            if ($success) {
                Write-Host "[完成] $model -> $targetDir (ModelScope)"
            } elseif ($Source -eq "modelscope") {
                Write-Error "[失败] $model 从 ModelScope 下载出错"
            } else {
                Write-Warning "ModelScope 下载失败，尝试 HuggingFace..."
            }
        }
    }

    if (-not $success -and ($Source -eq "huggingface" -or $Source -eq "auto")) {
        if ($hfRepo) {
            Write-Host "[下载] $model 从 HuggingFace ($hfRepo)..."
            $success = Download-ModelFromHuggingFace -RepoId $hfRepo -TargetDir $targetDir
            if ($success) {
                Write-Host "[完成] $model -> $targetDir (HuggingFace)"
            } else {
                Write-Error "[失败] $model 从 HuggingFace 下载出错"
            }
        }
    }
}
