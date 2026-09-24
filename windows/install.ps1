$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv-windows"

# 项目要求 Python 3.11。按以下顺序寻找解释器，取第一个版本号确实是 3.11 的：
#   1) 环境变量 JEV_PY311 指定的解释器
#   2) conda 已登记的环境（%USERPROFILE%\.conda\environments.txt）
#   3) PATH 里的 python
#   4) %USERPROFILE% 下的 anaconda3 / miniconda3（含 jev311 环境）
# 想直接指定就设 JEV_PY311，例如：
#   $env:JEV_PY311 = "C:\Path\To\python.exe"
function Test-Python311($Exe) {
    if (-not $Exe) { return $false }
    if (-not (Test-Path $Exe)) { return $false }
    $version = & $Exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    return $version -eq "3.11"
}

function Find-Python311 {
    if (Test-Python311 $env:JEV_PY311) { return $env:JEV_PY311 }

    $candidates = @()

    $envList = Join-Path $env:USERPROFILE ".conda\environments.txt"
    if (Test-Path $envList) {
        foreach ($line in (Get-Content $envList)) {
            $trimmed = $line.Trim()
            if ($trimmed) { $candidates += (Join-Path $trimmed "python.exe") }
        }
    }

    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) { $candidates += $fromPath.Source }

    foreach ($base in @(
        (Join-Path $env:USERPROFILE "anaconda3"),
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "AppData\Local\miniconda3")
    )) {
        $candidates += (Join-Path $base "envs\jev311\python.exe")
        $candidates += (Join-Path $base "python.exe")
    }

    foreach ($exe in $candidates) {
        if (Test-Python311 $exe) { return $exe }
    }
    return $null
}

$Py311 = Find-Python311
if (-not $Py311) {
    $message = '未找到 Python 3.11 解释器。请先创建一个：' + "`n" +
               '    conda create -n jev311 python=3.11 -y' + "`n" +
               '或指定已有解释器：' + "`n" +
               '    $env:JEV_PY311 = "C:\Path\To\python.exe"'
    throw $message
}

if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    & $Py311 -m venv $VenvPath
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "安装完成。运行：windows\start.ps1"

