#!/usr/bin/env pwsh
<#
    Covers the CUDA gate in the Scoop manifest on a host with no GPU.

    scoop-validate installs on a runner with no NVIDIA card, so post_install only ever
    takes the Vulkan branch there. This stubs nvidia-smi and asserts which build the
    gate picks, reading the script out of the manifest so the two cannot drift.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$manifestPath = Join-Path $PSScriptRoot '..' | Join-Path -ChildPath '..' | Join-Path -ChildPath 'bucket/lilbee.json'
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Past this line post_install downloads the CUDA build.
$lines = @($manifest.post_install)
$cut = [array]::IndexOf($lines, 'if ($build) {')
if ($cut -lt 0) { throw "could not find the download branch in post_install" }
$detect = ($lines[0..($cut - 1)] -join "`n")

$script:SmiOutput = @()
$script:SmiArgs = @()
# A function shadows the real executable, so Get-Command and & both resolve to this.
$stub = { $script:SmiArgs = $args; $script:SmiOutput }

$failures = 0

function Test-Gate {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string[]]$Output = @(),
        [Parameter(Mandatory)][AllowEmptyString()][string]$Expect,
        [switch]$NoSmi
    )
    $script:SmiOutput = $Output
    $script:SmiArgs = @()
    if ($NoSmi) {
        if (Test-Path function:nvidia-smi) { Remove-Item function:nvidia-smi }
    } else {
        Set-Item function:nvidia-smi $script:stub
    }

    $build = ''
    . ([scriptblock]::Create($script:detect))

    $shown = if ($build) { $build } else { 'vulkan' }
    if ($build -eq $Expect) {
        Write-Host "  ok   $Name -> $shown"
    } else {
        $want = if ($Expect) { $Expect } else { 'vulkan' }
        Write-Host "  FAIL ${Name}: expected $want, got $shown"
        $script:failures++
    }
}

Write-Host "CUDA gate, from $manifestPath"

# Driver floors from the CUDA toolkit release notes: 12.5 needs 555.85, 12.4 needs 551.61.
Test-Gate -Name 'R610 driver (610.88)' -Output @('610.88') -Expect 'cu125'
Test-Gate -Name 'CUDA 12.5 floor (555.85)' -Output @('555.85') -Expect 'cu125'
Test-Gate -Name 'just under the 12.5 floor (555.84)' -Output @('555.84') -Expect 'cu124'
Test-Gate -Name 'CUDA 12.4 floor (551.61)' -Output @('551.61') -Expect 'cu124'
Test-Gate -Name 'just under the 12.4 floor (551.60)' -Output @('551.60') -Expect ''
Test-Gate -Name 'pre-rename driver (550.54.14)' -Output @('550.54.14') -Expect ''
# A numeric-split comparison reads 551.7 as below 551.61. [version] does not.
Test-Gate -Name 'ordering: 551.7 is above 551.61' -Output @('551.7') -Expect 'cu124'
# --query-gpu prints one row per card.
Test-Gate -Name 'two cards' -Output @('610.88', '610.88') -Expect 'cu125'
# nvidia-smi present but answering nothing: driver installed, no card.
Test-Gate -Name 'no rows' -Output @() -Expect ''
Test-Gate -Name 'blank row' -Output @('') -Expect ''
Test-Gate -Name 'unparsable row' -Output @('N/A') -Expect ''
Test-Gate -Name 'no nvidia-smi at all' -Expect '' -NoSmi

# Locks the interface, not just the answer.
Set-Item function:nvidia-smi $stub
$script:SmiOutput = @('610.88')
$script:SmiArgs = @()
$build = ''
. ([scriptblock]::Create($detect))
if ($script:SmiArgs -join ' ' -match '--query-gpu=driver_version') {
    Write-Host "  ok   reads driver_version through --query-gpu"
} else {
    Write-Host "  FAIL expected --query-gpu=driver_version, got: $($script:SmiArgs -join ' ')"
    $failures++
}

# Every build the gate can pick must have a hash to verify it against.
$download = ($lines[$cut..($lines.Count - 1)] -join "`n")
foreach ($flavor in @('cu125', 'cu124')) {
    if ($download -match "'$flavor' = '[0-9a-f]{64}'") {
        Write-Host "  ok   $flavor has a pinned sha256"
    } else {
        Write-Host "  FAIL $flavor has no pinned sha256 in post_install"
        $failures++
    }
}

if ($failures -gt 0) { throw "$failures CUDA gate check(s) failed" }
Write-Host "all CUDA gate checks passed"
