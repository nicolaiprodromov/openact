<#
.SYNOPSIS
Regenerates openact/runtime/requirements-cu128.lock, the hash-locked packages of the model runtime.

.DESCRIPTION
Runs `uv pip compile --generate-hashes` for torch 2.10.0 (CUDA 12.8) and SemIf at the commit the installer
downloads (SEMIF_COMMIT in openact/server.py). The lock holds every dependency of both, for Windows and Linux.
SemIf itself is left out: the installer installs it from its SHA-256-checked archive, without dependencies.
Versions already in the lock are kept; delete a line to let uv choose that package again.
Needs uv 0.8.17, the version the installer pins.

.EXAMPLE
tools\make_locks.ps1 -Uv C:\path\to\uv.exe
#>
[CmdletBinding(PositionalBinding = $false)]
param([string]$Uv = "uv")

$version = & $Uv --version
if ($version -notmatch '^uv 0\.8\.17\b') { throw "Needs uv 0.8.17, the version the installer pins; found: $version" }
$lock = Join-Path (Split-Path $PSScriptRoot) "openact\runtime\requirements-cu128.lock"
$requirements = "torch==2.10.0`nsemif-phase1 @ https://github.com/TheoLeeCJ/SemIf-OpenJev/archive/23cf1f39fc9534fe81437200959b6dfc7106e45a.zip`n"
$requirements | & $Uv pip compile - --generate-hashes --universal --python-version 3.12 --torch-backend cu128 `
    --no-emit-package semif-phase1 --custom-compile-command "tools\make_locks.ps1" --output-file $lock
exit $LASTEXITCODE
