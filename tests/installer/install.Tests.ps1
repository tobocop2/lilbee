# Pester 5 tests for site/install.ps1: the build it picks for every machine case, and what it does with a download.

# Evaluated while Pester discovers the tests, so a Windows-only test can skip itself.
$onWindows = ($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows

BeforeAll {
    $env:LILBEE_INSTALL_LIB = '1'
    # Two-argument Join-Path: Windows PowerShell 5.1 takes no extra path segments.
    $script:InstallScript = Join-Path (Join-Path (Split-Path (Split-Path $PSScriptRoot)) 'site') 'install.ps1'
    . $script:InstallScript
    Set-StrictMode -Version Latest
    $script:TableDir = $PSScriptRoot
}

AfterAll {
    Remove-Item Env:LILBEE_INSTALL_LIB -ErrorAction SilentlyContinue
}

Describe 'machine cases' {
    It 'every machine case gets the build the case table names' {
        $cases = @(Import-Csv -Delimiter "`t" -LiteralPath (Join-Path $script:TableDir 'cases.tsv'))
        $sums = ConvertFrom-Checksum (Get-Content -LiteralPath (Join-Path $script:TableDir 'SHA256SUMS') -Raw)
        $shipped = @(Get-Content -LiteralPath (Join-Path $script:TableDir 'rocm.gfx.txt') | Where-Object { $_ })
        $cases.Count | Should -BeGreaterThan 100
        $wrong = @()
        $matched = 0
        foreach ($c in $cases) {
            try {
                $got = Select-LilbeeAsset $c.os $c.arch $c.avx2 $c.cuda $c.kfd $c.gfx $c.rocm_userland $c.variant $sums $shipped
            } catch {
                $got = 'ERROR'
            }
            if ($got -eq $c.expected) { $matched++ } else { $wrong += "$($c.id): expected $($c.expected), got $got" }
        }
        [Console]::Out.WriteLine("matched $matched of $($cases.Count)")
        $wrong | Should -BeNullOrEmpty
    }

    It 'an unknown variant is refused with the npm launcher text' {
        { Get-AssetName 'linux' 'x64' 'cu999' } | Should -Throw '*Unknown LILBEE_VARIANT "cu999". Valid: default, cu121, cu124, cu125, rocm, compat, compat-cu124, compat-rocm, or unset.*'
    }
}

Describe 'the install flow' {
    BeforeAll {
        function Publish-Asset {
            param([string]$Name, [string]$Body, [string]$Hash = '')
            $bytes = [Text.Encoding]::UTF8.GetBytes($Body)
            if (-not $Hash) { $Hash = ([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes)) -replace '-', '').ToLower() }
            $script:Release[$Name] = $Body
            $current = ''
            if ($script:Release.ContainsKey('SHA256SUMS')) { $current = $script:Release['SHA256SUMS'] }
            $script:Release['SHA256SUMS'] = $current + "$Hash  $Name`n"
        }
    }

    BeforeEach {
        $env:LILBEE_INSTALL_DIR = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $env:LILBEE_NO_MODIFY_PATH = '1'
        Remove-Item Env:LILBEE_RELEASE, Env:LILBEE_VARIANT -ErrorAction SilentlyContinue
        $script:Release = @{}
        $script:Urls = @()
        Mock Get-HostProfile { @{ Os = 'win32'; Arch = 'x64'; Avx2 = 'yes'; Cuda = '-'; Kfd = 'no'; Gfx = '-'; Userland = 'no' } }
        Mock Register-LilbeePath { }
        Mock Write-Say { }
        Mock Get-LilbeeFile {
            $script:Urls += $Url
            $name = $Url.Substring($Url.LastIndexOf('/') + 1)
            if (-not $script:Release.ContainsKey($name)) { throw "404 $name" }
            [IO.File]::WriteAllText($Path, $script:Release[$name])
        }
        $script:Dest = Join-Path $env:LILBEE_INSTALL_DIR 'lilbee.exe'
    }

    AfterEach {
        Remove-Item Env:LILBEE_INSTALL_DIR, Env:LILBEE_NO_MODIFY_PATH -ErrorAction SilentlyContinue
    }

    It 'installs a matching download' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'exe-bytes'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'exe-bytes'
        Should -Invoke Write-Say -ParameterFilter { $Message -like '*The SHA-256 matches SHA256SUMS*' }
        Should -Invoke Register-LilbeePath -Times 1 -ParameterFilter { $Folder -eq $env:LILBEE_INSTALL_DIR }
    }

    It 'refuses a download that does not match its SHA256SUMS line and installs nothing' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'tampered' ('0' * 64)
        { Install-Lilbee } | Should -Throw '*does not match SHA256SUMS. Nothing was installed.*'
        Test-Path -LiteralPath $script:Dest | Should -BeFalse
        @(Get-ChildItem -LiteralPath $env:LILBEE_INSTALL_DIR -Force -Filter '.lilbee.*').Count | Should -Be 0
    }

    It 'leaves the installed file as it was after a mismatch' {
        New-Item -ItemType Directory -Path $env:LILBEE_INSTALL_DIR | Out-Null
        [IO.File]::WriteAllText($script:Dest, 'old')
        Publish-Asset 'lilbee-windows-x86_64.exe' 'tampered' ('0' * 64)
        { Install-Lilbee } | Should -Throw
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'old'
    }

    It 'refuses a release without SHA256SUMS and names npm' {
        $script:Release['lilbee-windows-x86_64.exe'] = 'exe-bytes'
        { Install-Lilbee } | Should -Throw '*no SHA256SUMS file*npm install -g lilbee*'
        Test-Path -LiteralPath $script:Dest | Should -BeFalse
    }

    It 'refuses a SHA256SUMS that has no line for the asset' {
        Publish-Asset 'lilbee-macos-arm64' 'other'
        { Install-Lilbee } | Should -Throw '*This release has no build for win32/x64.*'
    }

    It 'does not download again when the installed file is current' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'current'
        New-Item -ItemType Directory -Path $env:LILBEE_INSTALL_DIR | Out-Null
        [IO.File]::WriteAllText($script:Dest, 'current')
        Install-Lilbee
        Should -Invoke Write-Say -ParameterFilter { $Message -like '*is up to date*' }
        @($script:Urls | Where-Object { $_ -like '*lilbee-windows-x86_64.exe' }).Count | Should -Be 0
    }

    It 'replaces an older file, keeps it as .old, and removes the .old on the next run' {
        New-Item -ItemType Directory -Path $env:LILBEE_INSTALL_DIR | Out-Null
        [IO.File]::WriteAllText($script:Dest, 'old')
        Publish-Asset 'lilbee-windows-x86_64.exe' 'new'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'new'
        Get-Content -LiteralPath "$($script:Dest).old" -Raw | Should -Be 'old'
        Install-Lilbee
        Test-Path -LiteralPath "$($script:Dest).old" | Should -BeFalse
    }

    It 'LILBEE_RELEASE pins the release the files come from' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'pinned'
        $env:LILBEE_RELEASE = 'v1.2.3'
        Install-Lilbee
        $script:Urls | Should -Contain 'https://github.com/tobocop2/lilbee/releases/download/v1.2.3/SHA256SUMS'
        $script:Urls | Should -Contain 'https://github.com/tobocop2/lilbee/releases/download/v1.2.3/lilbee-windows-x86_64.exe'
    }

    It 'without a pin the files come from the latest release' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'latest'
        Install-Lilbee
        $script:Urls | Should -Contain 'https://github.com/tobocop2/lilbee/releases/latest/download/SHA256SUMS'
    }

    It 'LILBEE_VARIANT names the build to install' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'plain'
        Publish-Asset 'lilbee-windows-x86_64-cu125.exe' 'cuda'
        $env:LILBEE_VARIANT = 'cu125'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'cuda'
    }

    It 'a GPU variant that the release does not ship falls back to the plain build' {
        Publish-Asset 'lilbee-windows-x86_64.exe' 'plain'
        $env:LILBEE_VARIANT = 'cu125'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'plain'
    }

    It 'a host without avx2 installs the compat build' {
        Mock Get-HostProfile { @{ Os = 'win32'; Arch = 'x64'; Avx2 = 'no'; Cuda = '-'; Kfd = 'no'; Gfx = '-'; Userland = 'no' } }
        Publish-Asset 'lilbee-windows-x86_64.exe' 'plain'
        Publish-Asset 'lilbee-compat-windows-x86_64.exe' 'compat'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'compat'
    }

    It 'a host whose driver reports CUDA 12.5 installs the cu125 build' {
        Mock Get-HostProfile { @{ Os = 'win32'; Arch = 'x64'; Avx2 = 'yes'; Cuda = '12.5'; Kfd = 'no'; Gfx = '-'; Userland = 'no' } }
        Publish-Asset 'lilbee-windows-x86_64.exe' 'plain'
        Publish-Asset 'lilbee-windows-x86_64-cu125.exe' 'cuda'
        Install-Lilbee
        Get-Content -LiteralPath $script:Dest -Raw | Should -Be 'cuda'
    }

    It 'Windows arm64 is refused before any download' {
        Mock Get-HostProfile { @{ Os = 'win32'; Arch = 'arm64'; Avx2 = 'yes'; Cuda = '-'; Kfd = 'no'; Gfx = '-'; Userland = 'no' } }
        { Install-Lilbee } | Should -Throw '*no standalone lilbee build for Windows arm64*'
        $script:Urls.Count | Should -Be 0
    }
}

Describe 'the PATH entry' {
    BeforeEach {
        $script:SavedPath = $env:Path
        $env:Path = 'C:\Windows'
        Remove-Item Env:LILBEE_NO_MODIFY_PATH, Env:GITHUB_ACTIONS, Env:GITHUB_PATH -ErrorAction SilentlyContinue
        Mock Add-UserPath { $true }
        Mock Write-Say { }
    }

    AfterEach {
        $env:Path = $script:SavedPath
        Remove-Item Env:LILBEE_NO_MODIFY_PATH, Env:GITHUB_ACTIONS, Env:GITHUB_PATH -ErrorAction SilentlyContinue
    }

    It 'adds a folder that is not on PATH to the user PATH' {
        Register-LilbeePath 'C:\Tools\lilbee'
        Should -Invoke Add-UserPath -Times 1 -ParameterFilter { $Folder -eq 'C:\Tools\lilbee' }
        $env:Path | Should -BeLike '*C:\Tools\lilbee'
    }

    It 'LILBEE_NO_MODIFY_PATH leaves the user PATH alone' {
        $env:LILBEE_NO_MODIFY_PATH = '1'
        Register-LilbeePath 'C:\Tools\lilbee'
        Should -Invoke Add-UserPath -Times 0
        Should -Invoke Write-Say -ParameterFilter { $Message -like '*add C:\Tools\lilbee to your PATH*' }
    }

    It 'a folder already on PATH needs no change' {
        $env:Path = 'C:\Windows;C:\Tools\lilbee'
        Register-LilbeePath 'C:\Tools\lilbee'
        Should -Invoke Add-UserPath -Times 0
    }

    It 'inside GitHub Actions the folder is appended to GITHUB_PATH' {
        $env:GITHUB_ACTIONS = 'true'
        $env:GITHUB_PATH = Join-Path $TestDrive 'github_path'
        $env:LILBEE_NO_MODIFY_PATH = '1'
        Register-LilbeePath 'C:\Tools\lilbee'
        (Get-Content -LiteralPath $env:GITHUB_PATH -Raw).Trim() | Should -Be 'C:\Tools\lilbee'
    }
}

Describe 'hardware detection' {
    BeforeAll {
        function Get-Banner {
            param([string]$Cuda)
            return "+-----------------------------+`n| NVIDIA-SMI 551.61    Driver Version: 551.61    CUDA Version: $Cuda     |`n"
        }
    }

    BeforeEach {
        $script:SavedEnv = @{ Arch = $env:PROCESSOR_ARCHITECTURE; Wow = $env:PROCESSOR_ARCHITEW6432; Root = $env:SystemRoot; Files = $env:ProgramFiles }
        # No drive letter: Join-Path refuses one on Linux, where the pod runs these tests.
        $env:SystemRoot = '/Windows'
        $env:ProgramFiles = '/Program Files'
        $env:PROCESSOR_ARCHITECTURE = 'AMD64'
        $env:PROCESSOR_ARCHITEW6432 = $null
        $script:MockAvx2 = 'yes'
        $script:MockCuda = '-'
        Mock Test-WindowsHost { $true }
        Mock Test-ProcessorFeature { if ($Feature -eq 40) { return ($script:MockAvx2 -eq 'yes') }; return $true }
        Mock Get-NvidiaBanner { if ($script:MockCuda -eq '-') { return '' }; return (Get-Banner $script:MockCuda) }
    }

    AfterEach {
        $env:PROCESSOR_ARCHITECTURE = $script:SavedEnv.Arch
        $env:PROCESSOR_ARCHITEW6432 = $script:SavedEnv.Wow
        $env:SystemRoot = $script:SavedEnv.Root
        $env:ProgramFiles = $script:SavedEnv.Files
    }

    It 'every Windows machine case: the facts read from the machine are the facts of its case row' {
        $cases = @(Import-Csv -Delimiter "`t" -LiteralPath (Join-Path $script:TableDir 'cases.tsv') | Where-Object { ($_.os -eq 'win32') -and ($_.variant -eq '') })
        $cases.Count | Should -BeGreaterThan 10
        $wrong = @()
        $matched = 0
        foreach ($c in $cases) {
            $env:PROCESSOR_ARCHITECTURE = 'AMD64'
            if ($c.arch -eq 'arm64') { $env:PROCESSOR_ARCHITECTURE = 'ARM64' }
            $script:MockAvx2 = $c.avx2
            $script:MockCuda = $c.cuda
            $facts = Get-HostProfile
            if ($c.arch -eq 'arm64') {
                $got = "$($facts.Os) $($facts.Arch)"
                $want = 'win32 arm64'
            } else {
                $got = "$($facts.Os) $($facts.Arch) $($facts.Avx2) $($facts.Cuda) $($facts.Kfd) $($facts.Gfx) $($facts.Userland)"
                $want = "$($c.os) $($c.arch) $($c.avx2) $($c.cuda) $($c.kfd) $($c.gfx) $($c.rocm_userland)"
            }
            if ($got -ceq $want) { $matched++ } else { $wrong += "$($c.id): expected $want, got $got" }
        }
        [Console]::Out.WriteLine("matched $matched of $($cases.Count)")
        $wrong | Should -BeNullOrEmpty
    }

    It 'a CPU that lacks processor feature 40 is read as no avx2' {
        Mock Test-ProcessorFeature { return ($Feature -ne 40) }
        (Get-HostProfile).Avx2 | Should -Be 'no'
        Should -Invoke Test-ProcessorFeature -Times 1 -Exactly
    }

    It 'a processor feature probe that fails counts as avx2 yes' {
        Mock Test-ProcessorFeature { throw 'no kernel32' }
        (Get-HostProfile).Avx2 | Should -Be 'yes'
    }

    It 'a 32-bit process on a 64-bit Windows reads the real architecture' {
        $env:PROCESSOR_ARCHITECTURE = 'x86'
        $env:PROCESSOR_ARCHITEW6432 = 'AMD64'
        (Get-HostProfile).Arch | Should -Be 'x64'
        $env:PROCESSOR_ARCHITEW6432 = 'ARM64'
        (Get-HostProfile).Arch | Should -Be 'arm64'
    }

    It 'reads the CUDA version from the banner, not the driver version' {
        $script:MockCuda = '12.4'
        (Get-HostProfile).Cuda | Should -Be '12.4'
    }

    It 'tries the System32 and the NVSMI paths when nvidia-smi is not on PATH' {
        Mock Get-NvidiaBanner { if ($Command -like '*NVSMI*') { return (Get-Banner '12.5') }; return '' }
        Get-CudaVersion | Should -Be '12.5'
        Mock Get-NvidiaBanner { if ($Command -like '*System32*') { return (Get-Banner '11.8') }; return '' }
        Get-CudaVersion | Should -Be '11.8'
    }

    It 'a banner without a CUDA version gives none' {
        Mock Get-NvidiaBanner { return 'NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.' }
        Get-CudaVersion | Should -Be '-'
    }

    It 'a machine that is not Windows is refused with the shell command' {
        Mock Test-WindowsHost { $false }
        { Get-HostProfile } | Should -Throw '*runs on Windows*install.sh*'
    }
}

Describe 'the user PATH write' {
    BeforeEach {
        $script:Events = [Collections.ArrayList]::new()
        $key = [pscustomobject]@{ State = @{ Value = 'C:\Old;%USERPROFILE%\bin'; Kind = [Microsoft.Win32.RegistryValueKind]::ExpandString } }
        # GetValue(name, default, options), GetValueKind(name) and SetValue(name, value, kind), as RegistryKey has them.
        $key | Add-Member -MemberType ScriptMethod -Name GetValue -Value { return $this.State.Value }
        $key | Add-Member -MemberType ScriptMethod -Name GetValueKind -Value { return $this.State.Kind }
        $key | Add-Member -MemberType ScriptMethod -Name SetValue -Value {
            $this.State.Value = $args[1]
            $this.State.Kind = $args[2]
            [void]$script:Events.Add('set')
        }
        $key | Add-Member -MemberType ScriptMethod -Name Close -Value { [void]$script:Events.Add('close') }
        $script:Key = $key
        Mock Open-UserEnvironmentKey { return $script:Key }
        Mock Send-EnvironmentChange { [void]$script:Events.Add('broadcast'); return $true }
    }

    It 'appends the folder, keeps the old entries and the value kind, then broadcasts the change' {
        Add-UserPath 'C:\Tools\lilbee' | Should -BeTrue
        $script:Key.State.Value | Should -Be 'C:\Old;%USERPROFILE%\bin;C:\Tools\lilbee'
        $script:Key.State.Kind | Should -Be ([Microsoft.Win32.RegistryValueKind]::ExpandString)
        $script:Events -join ',' | Should -Be 'set,broadcast,close'
    }

    It 'a folder that is already there, in any case and with a trailing slash, is not written or broadcast' {
        Add-UserPath 'c:\old\' | Should -BeFalse
        $script:Key.State.Value | Should -Be 'C:\Old;%USERPROFILE%\bin'
        $script:Events -join ',' | Should -Be 'close'
    }

    It 'an empty user Path becomes the folder alone, as an expandable string' {
        $script:Key.State.Value = ''
        Add-UserPath 'C:\Tools\lilbee' | Should -BeTrue
        $script:Key.State.Value | Should -Be 'C:\Tools\lilbee'
        $script:Key.State.Kind | Should -Be ([Microsoft.Win32.RegistryValueKind]::ExpandString)
    }

    It 'a registry kind of REG_SZ is kept' {
        $script:Key.State.Kind = [Microsoft.Win32.RegistryValueKind]::String
        Add-UserPath 'C:\Tools\lilbee' | Should -BeTrue
        $script:Key.State.Kind | Should -Be ([Microsoft.Win32.RegistryValueKind]::String)
    }

    # The call itself, not a mock: Windows only.
    It 'sends the change to every top-level window' -Skip:(-not $onWindows) {
        Send-EnvironmentChange | Should -BeTrue
    }
}

Describe 'the script file' {
    It 'is ASCII so Windows PowerShell 5.1 reads it without a byte order mark' {
        $bytes = [IO.File]::ReadAllBytes($script:InstallScript)
        $bytes.Length | Should -BeGreaterThan 1000
        @($bytes | Where-Object { $_ -gt 127 }).Count | Should -Be 0
    }

    It 'ends with the one top-level call' {
        $last = (Get-Content -LiteralPath $script:InstallScript | Where-Object { $_ })[-1]
        $last | Should -Be 'if (-not $env:LILBEE_INSTALL_LIB) { Install-Lilbee }'
    }
}
