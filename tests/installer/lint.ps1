# PSScriptAnalyzer over the installer and its tests; any finding fails.
# The default rules run, plus the syntax check for Windows PowerShell 5.1, which the installer must run on.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot)
$targets = @((Join-Path (Join-Path $root 'site') 'install.ps1'), (Join-Path $PSScriptRoot 'install.Tests.ps1'), $PSCommandPath)
$settings = @{
    Severity = @('Warning', 'Error')
    Rules    = @{ PSUseCompatibleSyntax = @{ Enable = $true; TargetVersions = @('5.1') } }
}
$findings = @($targets | ForEach-Object { Invoke-ScriptAnalyzer -Path $_ -Settings $settings })
$findings | Format-Table RuleName, Severity, ScriptName, Line, Message -AutoSize | Out-String -Width 200 | Write-Output
Write-Output "PSScriptAnalyzer: $($findings.Count) findings in $($targets.Count) files"
if ($findings.Count -gt 0) { exit 1 }
