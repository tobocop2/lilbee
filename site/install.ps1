# lilbee installer for Windows: picks the build for this machine, checks it
# against the release's SHA256SUMS, and installs one file.
#   irm https://lilbee.sh/install.ps1 | iex
# Settings: LILBEE_INSTALL_DIR, LILBEE_RELEASE, LILBEE_VARIANT, LILBEE_NO_MODIFY_PATH.

function Get-ReleasesUrl { return 'https://github.com/tobocop2/lilbee/releases' }

function Get-NpmHint { return "Install with npm (npm install -g lilbee) or download a binary from $(Get-ReleasesUrl)." }

function Write-Say {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'Installer messages go to the console.')]
    param([string]$Message)
    Write-Host "lilbee: $Message"
}

function Exit-Install {
    param([string]$Message)
    throw "lilbee: error: $Message"
}

# The CUDA build a driver ceiling such as 12.4 can run, or an empty string.
function Get-CudaBuild {
    param([string]$Cuda)
    if ($Cuda -eq '-') { return '' }
    $parts = $Cuda.Split('.')
    $ceiling = [int]$parts[0] * 100 + [int]$parts[1]
    if ($ceiling -ge 1205) { return 'cu125' }
    if ($ceiling -ge 1204) { return 'cu124' }
    if ($ceiling -ge 1201) { return 'cu121' }
    return ''
}

# The variant the hardware calls for.
function Get-HostVariant {
    param([string]$Os, [string]$Arch, [string]$Avx2, [string]$Cuda, [string]$Kfd, [string]$Gfx, [string]$Userland)
    $build = Get-CudaBuild $Cuda
    $compat = ($Arch -eq 'x64') -and ($Avx2 -eq 'no')
    $rocm = ($Os -eq 'linux') -and ($build -eq '') -and ($Kfd -eq 'yes') -and (($Gfx -ne '-') -or ($Userland -eq 'yes'))
    if ($Os -eq 'darwin') {
        if ($compat) { return 'compat' }
        return 'default'
    }
    if ($Os -eq 'win32') {
        if ($compat) {
            if (($build -eq 'cu124') -or ($build -eq 'cu125')) { return 'compat-cu124' }
            return 'compat'
        }
        if (($build -eq '') -or ($build -eq 'cu121')) { return 'default' }
        return $build
    }
    if ($compat) {
        if (($build -eq 'cu124') -or ($build -eq 'cu125')) { return 'compat-cu124' }
        if ($rocm) { return 'compat-rocm' }
        return 'compat'
    }
    if ($build -ne '') { return $build }
    if ($rocm) { return 'rocm' }
    return 'default'
}

# The release asset name for a host and variant; throws with the reason.
function Get-AssetName {
    param([string]$Os, [string]$Arch, [string]$Variant)
    $name = $Variant
    if ($name -eq 'default') { $name = '' }
    $variants = @('cu121', 'cu124', 'cu125', 'rocm', 'compat', 'compat-cu124', 'compat-rocm')
    if (($name -ne '') -and ($variants -notcontains $name)) {
        throw "Unknown LILBEE_VARIANT `"$Variant`". Valid: default, cu121, cu124, cu125, rocm, compat, compat-cu124, compat-rocm, or unset."
    }
    $compat = ($name -eq 'compat') -or $name.StartsWith('compat-')
    $prefix = 'lilbee'
    $gpu = $name
    if ($compat) {
        $prefix = 'lilbee-compat'
        $gpu = $name -replace '^compat-?', ''
    }
    $suffix = ''
    if ($gpu -ne '') { $suffix = "-$gpu" }
    switch ("$Os/$Arch") {
        'darwin/arm64' {
            if ($name -ne '') { throw 'macOS arm64 ships a single build; unset LILBEE_VARIANT.' }
            return 'lilbee-macos-arm64'
        }
        'darwin/x64' {
            if ($gpu -ne '') { throw 'macOS has no GPU-variant builds; unset LILBEE_VARIANT.' }
            return "$prefix-macos-x86_64"
        }
        'linux/x64' { return "$prefix-linux-x86_64$suffix" }
        'win32/x64' {
            if ($compat -and ($gpu -ne '') -and ($name -ne 'compat-cu124')) {
                throw 'Windows compat build has only the compat-cu124 GPU variant.'
            }
            return "$prefix-windows-x86_64$suffix.exe"
        }
    }
    throw "No standalone lilbee build for $Os/$Arch."
}

# True when every gfx target in the comma list is in the manifest.
function Test-GfxShipped {
    param([string]$Gfx, [string[]]$Shipped)
    if ($Shipped.Count -eq 0) { return $false }
    foreach ($target in $Gfx.Split(',')) {
        if ($Shipped -notcontains $target) { return $false }
    }
    return $true
}

# The asset to install. $Sums maps release asset names to their sha256.
function Select-LilbeeAsset {
    param(
        [string]$Os, [string]$Arch, [string]$Avx2, [string]$Cuda, [string]$Kfd, [string]$Gfx, [string]$Userland,
        [string]$Variant, [hashtable]$Sums, [string[]]$ShippedGfx
    )
    $hostGfx = $Gfx
    if ($Variant -ne '') {
        [void](Get-AssetName $Os $Arch $Variant)
        $wanted = $Variant
        $hostGfx = '-'
    } else {
        $wanted = Get-HostVariant $Os $Arch $Avx2 $Cuda $Kfd $Gfx $Userland
    }
    if ($wanted -eq 'default') { $wanted = '' }
    $baseline = 'default'
    if (($wanted -eq 'compat') -or $wanted.StartsWith('compat-')) { $baseline = 'compat' }
    $baseAsset = Get-AssetName $Os $Arch $baseline
    if (-not $Sums.ContainsKey($baseAsset)) { throw "This release has no build for $Os/$Arch." }
    if (($wanted -ne '') -and ($wanted -ne $baseline)) {
        $gpuAsset = Get-AssetName $Os $Arch $wanted
        if ($Sums.ContainsKey($gpuAsset)) {
            $shipped = $true
            if ($wanted.EndsWith('rocm') -and ($hostGfx -ne '-')) { $shipped = Test-GfxShipped $hostGfx $ShippedGfx }
            if ($shipped) { return $gpuAsset }
        }
    }
    return $baseAsset
}

# The sha256 lines of a SHA256SUMS text as a name -> hash table.
function ConvertFrom-Checksum {
    param([string]$Text)
    $sums = @{}
    foreach ($line in $Text -split "`n") {
        $fields = $line.Trim() -split '\s+'
        if ($fields.Count -eq 2) { $sums[$fields[1].TrimStart('*')] = $fields[0].ToLower() }
    }
    return $sums
}

function Get-LilbeeFile {
    param([string]$Url, [string]$Path)
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Path
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLower()
}

# What the NVIDIA tool at $Command prints, or an empty string when it cannot run.
function Get-NvidiaBanner {
    param([string]$Command)
    try {
        return ((& $Command 2>$null) -join "`n")
    } catch {
        return ''
    }
}

# The CUDA version the NVIDIA driver reports, such as 12.4, or '-'.
function Get-CudaVersion {
    $candidates = @('nvidia-smi', (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'), (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'))
    foreach ($command in $candidates) {
        if ((Get-NvidiaBanner $command) -match 'CUDA Version:\s*(\d+\.\d+)') { return $Matches[1] }
    }
    return '-'
}

function Test-WindowsHost {
    return ($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows
}

# True when the CPU reports processor feature $Feature (40 is AVX2).
function Test-ProcessorFeature {
    param([int]$Feature)
    if (-not ('Lilbee.Cpu' -as [type])) {
        Add-Type -Namespace Lilbee -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(int feature);'
    }
    return [Lilbee.Cpu]::IsProcessorFeaturePresent($Feature)
}

# The hardware of this Windows machine in the case-table vocabulary.
function Get-HostProfile {
    if (-not (Test-WindowsHost)) {
        Exit-Install 'this installer runs on Windows. On macOS and Linux, run: curl -fsSL https://lilbee.sh/install.sh | sh'
    }
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
    $facts = @{ Os = 'win32'; Arch = 'x64'; Avx2 = 'yes'; Cuda = '-'; Kfd = 'no'; Gfx = '-'; Userland = 'no' }
    if ($arch -eq 'ARM64') { $facts.Arch = 'arm64' } elseif ($arch -ne 'AMD64') { $facts.Arch = $arch }
    if ($facts.Arch -ne 'x64') { return $facts }
    try {
        if (-not (Test-ProcessorFeature 40)) { $facts.Avx2 = 'no' }
    } catch {
        $facts.Avx2 = 'yes'
    }
    $facts.Cuda = Get-CudaVersion
    return $facts
}

function Open-UserEnvironmentKey {
    return [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
}

# Tells running programs, Explorer included, that the user environment changed. Returns $false when it could not.
function Send-EnvironmentChange {
    try {
        if (-not ('Lilbee.Broadcast' -as [type])) {
            Add-Type -Namespace Lilbee -Name Broadcast -MemberDefinition '[DllImport("user32.dll", CharSet = CharSet.Auto)] public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint msg, UIntPtr wParam, string lParam, uint flags, uint timeout, out UIntPtr result);'
        }
        $result = [UIntPtr]::Zero
        $sent = [Lilbee.Broadcast]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result)
        return ($sent -ne [IntPtr]::Zero)
    } catch {
        return $false
    }
}

# Adds the folder to the user Path, kept as REG_EXPAND_SZ so %VARIABLES% in it still expand.
# Then broadcasts the change, so new terminals started from Explorer see the folder.
function Add-UserPath {
    param([string]$Folder)
    $key = Open-UserEnvironmentKey
    try {
        $current = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
        if ($current -ne '') { $kind = $key.GetValueKind('Path') }
        $entries = @($current -split ';' | Where-Object { $_ -ne '' })
        if ($entries | Where-Object { $_.TrimEnd('\') -ieq $Folder.TrimEnd('\') }) { return $false }
        $key.SetValue('Path', (($entries + $Folder) -join ';'), $kind)
        [void](Send-EnvironmentChange)
        return $true
    } finally {
        $key.Close()
    }
}

function Register-LilbeePath {
    param([string]$Folder)
    if (($env:GITHUB_ACTIONS -eq 'true') -and $env:GITHUB_PATH) {
        Add-Content -LiteralPath $env:GITHUB_PATH -Value $Folder -Encoding Ascii
    }
    if (($env:Path -split ';') -contains $Folder) { return }
    if ($env:LILBEE_NO_MODIFY_PATH -eq '1') {
        Write-Say "add $Folder to your PATH."
        return
    }
    if (Add-UserPath $Folder) { Write-Say "added $Folder to your user PATH. Open a new terminal to use lilbee." }
    $env:Path = "$env:Path;$Folder"
}

function Test-FreeSpace {
    param([string]$Folder)
    try {
        $drive = New-Object System.IO.DriveInfo ([IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Folder).Path))
        return $drive.AvailableFreeSpace -ge 2GB
    } catch {
        return $true
    }
}

function Install-LilbeeFile {
    param([string]$Url, [string]$Dest, [string]$Expected, [string]$Asset)
    $folder = Split-Path -Parent $Dest
    if (-not (Test-FreeSpace $folder)) { Exit-Install "not enough free space in ${folder}: lilbee needs about 2 GB." }
    $partial = Join-Path $folder ".lilbee.$([guid]::NewGuid().ToString('N')).part"
    Write-Say "downloading $Asset"
    try {
        Get-LilbeeFile $Url $partial
        if ((Get-Sha256 $partial) -ne $Expected) {
            Exit-Install "the SHA-256 of $Asset does not match SHA256SUMS. Nothing was installed."
        }
        if (Test-Path -LiteralPath $Dest) { Move-Item -LiteralPath $Dest -Destination "$Dest.old" -Force }
        Move-Item -LiteralPath $partial -Destination $Dest -Force
    } finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    }
    Write-Say "installed $Dest ($Asset). The SHA-256 matches SHA256SUMS."
}

function Install-Lilbee {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }
    $facts = Get-HostProfile
    $npmHint = Get-NpmHint
    if ($facts.Arch -ne 'x64') { Exit-Install "no standalone lilbee build for Windows $($facts.Arch). $npmHint" }
    $forced = $env:LILBEE_VARIANT
    if (-not $forced) { $forced = '' }
    $baseUrl = "$(Get-ReleasesUrl)/latest/download"
    if ($env:LILBEE_RELEASE) { $baseUrl = "$(Get-ReleasesUrl)/download/$($env:LILBEE_RELEASE)" }
    $work = Join-Path ([IO.Path]::GetTempPath()) "lilbee-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        $sumsFile = Join-Path $work 'SHA256SUMS'
        try { Get-LilbeeFile "$baseUrl/SHA256SUMS" $sumsFile } catch {
            Exit-Install "this release has no SHA256SUMS file, so lilbee cannot verify a download. $npmHint"
        }
        $sums = ConvertFrom-Checksum (Get-Content -LiteralPath $sumsFile -Raw)
        try {
            $asset = Select-LilbeeAsset $facts.Os $facts.Arch $facts.Avx2 $facts.Cuda $facts.Kfd $facts.Gfx $facts.Userland $forced $sums @()
        } catch {
            Exit-Install "$($_.Exception.Message) $npmHint"
        }
        $expected = $sums[$asset]
        $folder = $env:LILBEE_INSTALL_DIR
        if (-not $folder) { $folder = Join-Path $env:LOCALAPPDATA 'Programs\lilbee' }
        $dest = Join-Path $folder 'lilbee.exe'
        New-Item -ItemType Directory -Force -Path $folder | Out-Null
        Remove-Item -LiteralPath "$dest.old" -Force -ErrorAction SilentlyContinue
        if ((Test-Path -LiteralPath $dest) -and ((Get-Sha256 $dest) -eq $expected)) {
            Write-Say "lilbee is up to date at $dest ($asset)."
        } else {
            Install-LilbeeFile "$baseUrl/$asset" $dest $expected $asset
        }
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
    Register-LilbeePath $folder
    $other = Get-Command lilbee -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($other -and ($other.Source -ne $dest)) { Write-Warning "another lilbee comes first on PATH: $($other.Source)" }
    Write-Say "to uninstall, delete $dest. Models and data stay in place."
}

if (-not $env:LILBEE_INSTALL_LIB) { Install-Lilbee }
