<#
.SYNOPSIS
Runs the OpenAct test suites and prints a summary table.

.DESCRIPTION
- *_test.py suites (pure Python) run under every CPython 3.11+ found (PATH, py -3.11, py -3.13),
  and always inside Blender 4.5 (Python 3.11) and 5.1 (Python 3.13).
- *_blender.py suites run on every Blender in -Versions, as blender -b --factory-startup --python.
- e2e_blender.py runs only with -Model: build, noaddon and reopen in a temporary folder, with the
  real model on -Gpu and -Port (a GPU lease: port 18765 on GPU 0, 18767 on GPU 1). Each build
  needs the port free and 9 GB free on the GPU; the runner waits up to 30 minutes for the memory.
A run that takes longer than 10 minutes (30 for an e2e build) is stopped. Every suite ends with a
RESULT line. The script exits with 1 if any run failed.

.EXAMPLE
tests\run_all.ps1 -Model -Versions 4.5,5.1 -Gpu 1
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Model,
    [string[]]$Versions = @("4.5", "5.0", "5.1"),
    [int]$Gpu = 0,
    [int]$Port = 18765 + 2 * $Gpu
)

# "-Versions 4.5,5.0,5.1" arrives as numbers (5.0 becomes "5"), or as one string under -File.
$Versions = @($Versions -split '[,\s]+' -ne '' -replace '^\d+$', '$0.0')
$Rows = New-Object System.Collections.Generic.List[object]

function Add-Row($Suite, $Where, $Ok, $Pass, $Fail, $Seconds, $Detail) {
    $Rows.Add([pscustomobject]@{
        Suite = $Suite; Where = $Where; Result = $(if ($Ok) { "PASS" } else { "FAIL" })
        Pass = $Pass; Fail = $Fail; Seconds = [math]::Round($Seconds, 1); Detail = $Detail
    })
}

function Invoke-Suite([string]$Suite, [string]$Where, [string]$Exe, [string[]]$Arguments, [int]$Minutes = 10) {
    Write-Host ""
    Write-Host "=== $Suite ($Where)" -ForegroundColor Cyan
    $start = [Diagnostics.ProcessStartInfo]::new($Exe, ($Arguments | ForEach-Object { '"' + $_ + '"' }) -join " ")
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    $start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"  # else CPython writes the ANSI code page into the pipe
    $start.EnvironmentVariables["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the server's --gpu is the card nvidia-smi -i reads
    $pass = $fail = 0
    $result = $null
    $timedOut = $true
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($start)
    $errors = $process.StandardError.ReadToEndAsync()  # drained alongside, so a full stderr pipe cannot stall the suite
    $next = $process.StandardOutput.ReadLineAsync()
    while ($clock.Elapsed.TotalMinutes -lt $Minutes) {
        if (-not $next.Wait(1000)) { continue }
        $line = $next.Result
        if ($null -eq $line) { $timedOut = $false; break }
        $next = $process.StandardOutput.ReadLineAsync()
        Write-Host $line
        if ($line.StartsWith("PASS ")) { $pass++ } elseif ($line.StartsWith("FAIL ")) { $fail++ }
        if ($line -match '^(?:BLENDER (.+)|PYTHON \S+) RESULT (OK|FAILED: .*)$') {
            $result = $Matches[2]
            if ($Matches[1]) { $Where = "Blender " + $Matches[1] }
        }
    }
    if ($timedOut -and -not $process.HasExited) { $process.Kill() }  # a model server exits when its Blender is gone
    $process.WaitForExit()
    if ($errors.Result.Trim()) { Write-Host $errors.Result.TrimEnd() }
    $code = $process.ExitCode
    $ok = -not $timedOut -and $result -eq "OK" -and $code -eq 0
    $detail = if ($ok) { "" } elseif ($timedOut) { "timeout after $Minutes min" } elseif (-not $result) { "no RESULT line (exit code $code)" } elseif ($result -eq "OK") { "exit code $code" } else { $result }
    Add-Row $Suite $Where $ok $pass $fail $clock.Elapsed.TotalSeconds $detail
}

function Find-CPython {
    $code = "import sys; print('%d.%d.%d' % sys.version_info[:3]); print(sys.executable)"
    $found = [ordered]@{}
    foreach ($probe in { python -c $code }, { py -3.11 -c $code }, { py -3.13 -c $code }) {
        $out = @(try { & $probe 2>$null } catch { })
        $version = if ($out.Count -eq 2) { $out[0] -as [version] }
        if ($version -and $version -ge [version]"3.11" -and (Test-Path $out[1])) {
            $found[$out[1]] = [pscustomobject]@{ Exe = $out[1]; Version = $out[0] }
        }
    }
    $found.Values
}

function Get-Blender([string]$Version) {
    $exe = "C:\Program Files\Blender Foundation\Blender $Version\blender.exe"
    if (Test-Path $exe) { return $exe }
    Add-Row "-" "Blender $Version" $false 0 0 0 "not found: $exe"
}

$saved = [Console]::OutputEncoding
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("openact-e2e-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
try {
    try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }  # redirected output keeps the suites' UTF-8; no console, no change
    $e2e = if ($Model) { "e2e with the real model on GPU $Gpu, port $Port" } else { "no e2e (needs -Model)" }
    Write-Host "OpenAct tests: Blender $($Versions -join ', '); $e2e"

    $files = @(Get-ChildItem -Path $PSScriptRoot -File | Sort-Object Name)
    $pure = @($files | Where-Object { $_.Name -like "*_test.py" })
    $suites = @($files | Where-Object { $_.Name -like "*_blender.py" -and $_.Name -ne "e2e_blender.py" })
    if ($pure) {
        $pythons = @(Find-CPython)
        if (-not $pythons) { Write-Host "No CPython 3.11+ found: the pure suites run inside Blender only." }
        foreach ($python in $pythons) {
            foreach ($suite in $pure) { Invoke-Suite $suite.Name "Python $($python.Version)" $python.Exe @($suite.FullName) }
        }
        foreach ($version in "4.5", "5.1") {  # Python 3.11 and 3.13, whatever -Versions holds
            $exe = Get-Blender $version
            if ($exe) { foreach ($suite in $pure) { Invoke-Suite $suite.Name "Blender $version" $exe @("-b", "--factory-startup", "--python", $suite.FullName) } }
        }
    }
    foreach ($version in $Versions) {
        $exe = Get-Blender $version
        if (-not $exe) { continue }
        foreach ($suite in $suites) { Invoke-Suite $suite.Name "Blender $version" $exe @("-b", "--factory-startup", "--python", $suite.FullName) }
        if (-not $Model) { continue }
        # The build starts a model server: it must not attach to (and later shut down) a server already on the
        # port, and it needs 9 GB on a GPU that other jobs may be using.
        $busy = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)[0]
        $free = $null
        for ($waited = 0; -not $busy; $waited += 30) {
            $read = try { "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $Gpu 2>$null)" } catch { "" }
            $free = if ($read -match '^\s*(\d+)\s*$') { [int]$Matches[1] }
            if ($null -eq $free -or $free -ge 9216 -or $waited -ge 1800) { break }
            Write-Host ("GPU {0}: {1:N1} GB free, waiting for 9 GB" -f $Gpu, ($free / 1024))
            Start-Sleep -Seconds 30
        }
        $blocked = if ($busy) { "port $Port in use by $((Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue).Name) (pid $($busy.OwningProcess))" }
            elseif ($null -eq $free) { "nvidia-smi cannot read GPU $Gpu" }
            elseif ($free -lt 9216) { "GPU ${Gpu}: {0:N1} GB free, needs 9 GB" -f ($free / 1024) }
        if ($blocked) { Add-Row "e2e_blender.py" "Blender $version" $false 0 0 0 $blocked; continue }
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $blend = Join-Path $tmp "e2e-$version.blend"
        foreach ($phase in "build", "noaddon", "reopen") {
            $argv = @("-b", "--factory-startup", "--python", (Join-Path $PSScriptRoot "e2e_blender.py"), "--", $phase, $blend)
            if ($phase -eq "build") { Invoke-Suite "e2e_blender.py build" "Blender $version" $exe ($argv + "$Port", "$Gpu") 30 }
            else { Invoke-Suite "e2e_blender.py $phase" "Blender $version" $exe $argv }
        }
    }

    Write-Host ""
    Write-Host ($Rows | Format-Table -AutoSize | Out-String -Width 1000).Trim()
    $failed = @($Rows | Where-Object { $_.Result -ne "PASS" }).Count
    Write-Host "$($Rows.Count - $failed) of $($Rows.Count) runs passed."
}
finally {
    try { [Console]::OutputEncoding = $saved } catch { }
    Remove-Item -Recurse -Force -Path $tmp -ErrorAction SilentlyContinue
}
exit [int]($failed -gt 0)
