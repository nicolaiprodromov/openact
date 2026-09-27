<#
.SYNOPSIS
Runs the OpenAct test suites and prints a summary table.

.DESCRIPTION
- *_test.py suites (pure Python) run under every CPython 3.11+ found (PATH, py -3.11, py -3.13),
  and always inside Blender 4.5 (Python 3.11) and 5.1 (Python 3.13).
- *_blender.py suites run on every Blender in -Versions, as blender -b --factory-startup --python.
  lookup_blender.py runs with -- --bench: its main-thread budgets are merge gates. Following the
  design's timing rule, a result above its budget prints a WARN line, counted in the summary, and
  fails only above twice the budget.
- e2e_blender.py runs only with -Model: build, noaddon, reopen and reopen-bumped in a temporary folder, with
  the real model on -Gpu and -Port (a GPU lease: port 18765 on GPU 0, 18767 on GPU 1). Each build
  needs the port free and 9 GB free on the GPU; the runner waits up to 30 minutes for the memory.
A run that takes longer than 10 minutes (30 for an e2e build) is stopped. Every suite ends with a
RESULT line. The script exits with 1 if any run failed.
- Every suite runs at BelowNormal priority, with OMP_NUM_THREADS and MKL_NUM_THREADS set to -Threads (4), and
  every Blender with -t (-Threads), so a run leaves most of the machine to its user.

.EXAMPLE
tests\run_all.ps1 -Model -Versions 4.5,5.1 -Gpu 1
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Model,
    [string[]]$Versions = @("4.5", "5.0", "5.1"),
    [int]$Gpu = 0,
    [int]$Port = 18765 + 2 * $Gpu,
    [int]$Threads = 4
)

# "-Versions 4.5,5.0,5.1" arrives as numbers (5.0 becomes "5"), or as one string under -File.
$Versions = @($Versions -split '[,\s]+' -ne '' -replace '^\d+$', '$0.0')
$Rows = New-Object System.Collections.Generic.List[object]

# Each run goes into a job object. The kernel adds everything the suite starts, so ending the job never
# depends on parent PIDs (which Windows reuses) and still reaches orphans. Kill-on-close ends the suite's
# processes when its handle closes, including when the runner itself dies.
Add-Type -Namespace OpenAct -Name Job -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] struct Limits {  // JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    public long PerProcessUserTime, PerJobUserTime; public uint Flags; public UIntPtr MinWorkingSet, MaxWorkingSet;
    public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass, SchedulingClass;
    public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemory, PeakJobMemory;
}
[DllImport("kernel32.dll")] static extern IntPtr CreateJobObject(IntPtr attributes, IntPtr name);
[DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref Limits info, int size);
[DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
[DllImport("kernel32.dll")] static extern bool QueryInformationJobObject(IntPtr job, int infoClass, long[] info, int size, IntPtr returned);
[DllImport("kernel32.dll")] static extern bool TerminateJobObject(IntPtr job, uint exitCode);
[DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
// A zero handle means "the caller's own job" to the information calls, so every method refuses it.
public static IntPtr Contain(IntPtr process) {  // zero if the suite could not be put in a job
    IntPtr job = CreateJobObject(IntPtr.Zero, IntPtr.Zero);
    if (job == IntPtr.Zero) return job;
    Limits limits = new Limits { Flags = 0x2000 };  // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    SetInformationJobObject(job, 9, ref limits, Marshal.SizeOf(limits));  // JobObjectExtendedLimitInformation
    if (AssignProcessToJobObject(job, process)) return job;
    CloseHandle(job);
    return IntPtr.Zero;
}
public static void End(IntPtr job) { if (job != IntPtr.Zero) TerminateJobObject(job, 1); }
public static int Leftovers(IntPtr job) {
    // JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: ActiveProcesses is the DWORD at byte 40. Processes that end on their
    // own within a second of the suite (console hosts, launcher stubs) are not leftovers.
    long[] info = new long[6];
    int started = Environment.TickCount;
    while (job != IntPtr.Zero && QueryInformationJobObject(job, 1, info, 48, IntPtr.Zero)) {
        int active = (int)info[5];
        if (active == 0 || Environment.TickCount - started >= 1000) return active;
        System.Threading.Thread.Sleep(50);
    }
    return 0;
}
'@

function Add-Row($Suite, $Where, $Ok, $Pass, $Fail, $Seconds, $Detail) {
    $Rows.Add([pscustomobject]@{
        Suite = $Suite; Where = $Where; Result = $(if ($Ok) { "PASS" } else { "FAIL" })
        Pass = $Pass; Fail = $Fail; Seconds = [math]::Round($Seconds, 1); Detail = $Detail
    })
}

function Invoke-Suite([string]$Suite, [string]$Where, [string]$Exe, [string[]]$Arguments, [int]$Minutes = 10) {
    Write-Host ""
    Write-Host "=== $Suite ($Where)" -ForegroundColor Cyan
    if ((Split-Path -Leaf $Exe) -ieq "blender.exe") { $Arguments = @("-t", "$Threads") + $Arguments }
    $start = [Diagnostics.ProcessStartInfo]::new($Exe, ($Arguments | ForEach-Object { '"' + $_ + '"' }) -join " ")
    $start.UseShellExecute = $false
    $start.WorkingDirectory = Split-Path $PSScriptRoot  # the repository root, like the README's per-suite commands
    $start.RedirectStandardOutput = $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    $start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"  # else CPython writes the ANSI code page into the pipe
    $start.EnvironmentVariables["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the server's --gpu is the card nvidia-smi -i reads
    $start.EnvironmentVariables["OMP_NUM_THREADS"] = $start.EnvironmentVariables["MKL_NUM_THREADS"] = "$Threads"
    # Enabling or disabling an add-on re-syncs the extension wheels of Blender's user folder to the session's
    # enabled extensions (none under --factory-startup), which would delete the user's own extension packages.
    $start.EnvironmentVariables["BLENDER_USER_RESOURCES"] = Join-Path $userRoot $Where.Replace(" ", "-")
    $pass = $fail = $warn = $leftover = 0
    $result = $endedAt = $null
    $timedOut = $false
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($start)
    $job = [OpenAct.Job]::Contain($process.Handle)
    $streams = $process.StandardOutput, $process.StandardError
    # Both pipes are read line by line, so a traceback on stderr shows even while a descendant holds the pipe open.
    $reads = @($streams[0].ReadLineAsync(), $streams[1].ReadLineAsync())
    while ($true) {
        $open = @($reads -ne $null)
        if ($open) { $null = [Threading.Tasks.Task]::WaitAny([Threading.Tasks.Task[]]$open, 1000) } else { $null = $process.WaitForExit(1000) }
        foreach ($k in 0, 1) {
            if (-not $reads[$k] -or -not $reads[$k].IsCompleted) { continue }
            $line = $reads[$k].Result
            $reads[$k] = if ($null -ne $line) { $streams[$k].ReadLineAsync() }
            if ($null -eq $line) { continue }
            Write-Host $line
            if ($k -eq 1) { continue }
            if ($line.StartsWith("PASS ")) { $pass++ } elseif ($line.StartsWith("FAIL ")) { $fail++ } elseif ($line.StartsWith("WARN ")) { $warn++ }
            if ($line -match '^(?:BLENDER (.+)|PYTHON \S+) RESULT (OK|FAILED: .*)$') {
                $result = $Matches[2]
                if ($Matches[1]) { $Where = "Blender " + $Matches[1] }
            }
        }
        if ($null -eq $endedAt) {
            # End the job once the suite has exited (whatever it left running) or at the deadline (everything).
            if ($process.HasExited) { $leftover = [OpenAct.Job]::Leftovers($job) }
            elseif ($clock.Elapsed.TotalMinutes -ge $Minutes) { $timedOut = $true }
            else { continue }
            [OpenAct.Job]::End($job)
            $endedAt = $clock.Elapsed.TotalSeconds
        }
        if ($process.HasExited -and -not ($reads[0] -or $reads[1])) { break }
        if ($clock.Elapsed.TotalSeconds - $endedAt -ge 5) {
            if ($process.HasExited) { break }  # a pipe is held by a process outside the job
            $process.Kill()  # the suite is not in a job (Contain returned zero)
            $endedAt = $clock.Elapsed.TotalSeconds
        }
    }
    $null = [OpenAct.Job]::CloseHandle($job)
    $code = $process.ExitCode
    $ok = -not $timedOut -and $result -eq "OK" -and $code -eq 0
    $detail = if ($ok) { "" } elseif ($timedOut) { "timeout after $Minutes min" } elseif (-not $result) { "no RESULT line (exit code $code)" } elseif ($result -eq "OK") { "exit code $code" } else { $result }
    if ($leftover) { $detail = (@($detail, "leftover processes ended: $leftover") -ne "") -join "; " }
    if ($warn) { $detail = (@($detail, "warnings: $warn") -ne "") -join "; " }
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
$userRoot = Join-Path $tmp "blender-user"  # Blender user folders for this run only, removed with $tmp
try {
    (Get-Process -Id $PID).PriorityClass = "BelowNormal"  # every suite and its children inherit it
    New-Item -ItemType Directory -Force -Path $userRoot | Out-Null
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
        foreach ($suite in $suites) {
            $name = $suite.Name
            $argv = @("-b", "--factory-startup", "--python", $suite.FullName)
            if ($name -eq "lookup_blender.py") { $name += " --bench"; $argv += "--", "--bench" }
            Invoke-Suite $name "Blender $version" $exe $argv
        }
        if (-not $Model) { continue }
        # The build starts a model server: it must not attach to (and later shut down) a server already on the
        # port, and it needs 9 GB on a GPU that other jobs may be using.
        for ($waited = 0; ; $waited += 30) {
            $busy = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)[0]
            $read = try { "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $Gpu 2>$null)" } catch { "" }
            $free = if ($read -match '^\s*(\d+)\s*$') { [int]$Matches[1] }
            if ($busy -or $null -eq $free -or $free -ge 9216 -or $waited -ge 1800) { break }
            Write-Host ("GPU {0}: {1:N1} GB free, waiting for 9 GB" -f $Gpu, ($free / 1024))
            Start-Sleep -Seconds 30
        }
        $blocked = if ($busy) { "port $Port in use by $((Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue).Name) (pid $($busy.OwningProcess))" }
            elseif ($null -eq $free) { "nvidia-smi cannot read GPU $Gpu" }
            elseif ($free -lt 9216) { "GPU ${Gpu}: {0:N1} GB free, needs 9 GB" -f ($free / 1024) }
        if ($blocked) { Add-Row "e2e_blender.py" "Blender $version" $false 0 0 0 $blocked; continue }
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $blend = Join-Path $tmp "e2e-$version.blend"
        foreach ($phase in "build", "noaddon", "reopen", "reopen-bumped") {
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
