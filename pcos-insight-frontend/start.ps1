param(
    [string]$Artifacts = (Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts'),
    [string]$Python = (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'),
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Python environment not found. Pass -Python with the interpreter used for notebook export.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Artifacts 'manifest.json'))) {
    throw 'Artifacts not found. Pass -Artifacts with the exact folder printed by your notebook export.'
}
& $Python (Join-Path $PSScriptRoot 'backend\server.py') --artifacts $Artifacts --port $Port
