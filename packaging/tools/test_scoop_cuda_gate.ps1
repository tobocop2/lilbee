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

# Past this line post_install downloads 737 MB.
$lines = @($manifest.post_install)
$cut = [array]::IndexOf($lines, 'if ($useCuda) {')
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
        [Parameter(Mandatory)][bool]$Expect,
        [switch]$NoSmi
    )
    $script:SmiOutput = $Output
    $script:SmiArgs = @()
    if ($NoSmi) {
        if (Test-Path function:nvidia-smi) { Remove-Item function:nvidia-smi }
    } else {
        Set-Item function:nvidia-smi $script:stub
    }

    $useCuda = $false
    . ([scriptblock]::Create($script:detect))

    if ($useCuda -eq $Expect) {
        Write-Host "  ok   $Name (useCuda=$useCuda)"
    } else {
        Write-Host "  FAIL ${Name}: expected useCuda=$Expect, got $useCuda"
        $script:failures++
    }
}

Write-Host "CUDA gate, from $manifestPath"

# R610 reports CUDA 13.3 and must reach cu125.
Test-Gate -Name 'R610 driver (610.88)' -Output @('610.88') -Expect $true
# The Windows floor for CUDA 12.5, and one step under it.
Test-Gate -Name 'CUDA 12.5 floor (555.85)' -Output @('555.85') -Expect $true
Test-Gate -Name 'just below the floor (554.99)' -Output @('554.99') -Expect $false
Test-Gate -Name 'pre-rename driver (550.54.14)' -Output @('550.54.14') -Expect $false
# --query-gpu prints one row per card.
Test-Gate -Name 'two cards' -Output @('610.88', '610.88') -Expect $true
# nvidia-smi present but answering nothing: driver installed, no card.
Test-Gate -Name 'no rows' -Output @() -Expect $false
Test-Gate -Name 'blank row' -Output @('') -Expect $false
Test-Gate -Name 'no nvidia-smi at all' -Expect $false -NoSmi

# Locks the interface, not just the answer.
Set-Item function:nvidia-smi $stub
$script:SmiOutput = @('610.88')
$script:SmiArgs = @()
$useCuda = $false
. ([scriptblock]::Create($detect))
if ($script:SmiArgs -join ' ' -match '--query-gpu=driver_version') {
    Write-Host "  ok   reads driver_version through --query-gpu"
} else {
    Write-Host "  FAIL expected --query-gpu=driver_version, got: $($script:SmiArgs -join ' ')"
    $failures++
}

if ($failures -gt 0) { throw "$failures CUDA gate check(s) failed" }
Write-Host "all CUDA gate checks passed"
