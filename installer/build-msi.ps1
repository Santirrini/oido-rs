# build-msi.ps1
# Set error action to stop so it fails early on any error
$ErrorActionPreference = "Stop"

# Navigate to root directory (if script is run from installer/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($null -eq $scriptDir -or $scriptDir -eq "") { $scriptDir = "." }
Push-Location $scriptDir\..

# 1. Clean / create folders
Remove-Item -Path "installer/staging" -Recurse -ErrorAction SilentlyContinue | Out-Null
Remove-Item -Path "installer/dist" -Recurse -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "installer/staging" -Force | Out-Null
New-Item -ItemType Directory -Path "installer/dist" -Force | Out-Null

# 2. Build cargo in release with updater feature and static CRT
Write-Host "Building oido (static CRT)..."
$env:CMAKE_MSVC_RUNTIME_LIBRARY = "MultiThreaded"
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --release -p oido --features updater
Remove-Item env:CMAKE_MSVC_RUNTIME_LIBRARY -ErrorAction SilentlyContinue
Remove-Item env:RUSTFLAGS -ErrorAction SilentlyContinue

# 3. Copy executable to staging
Copy-Item -Path "target/release/oido.exe" -Destination "installer/staging/oido.exe" -Force

# 4. Extract version and validate
$version = (Get-Content "Cargo.toml" | Select-String '^\s*version\s*=\s*"([^"]+)"' | ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -First 1).Trim()
if ([string]::IsNullOrWhiteSpace($version)) {
    throw "Failed to extract version from Cargo.toml"
}
Write-Host "Version extracted: $version"

# 5. Compile WiX installer
Write-Host "Compiling installer with candle..."
& candle.exe "-dVersion=$version" -o "installer/staging/" "installer/oido.wxs"
if ($LastExitCode -ne 0) { throw "candle.exe failed with exit code $LastExitCode" }

Write-Host "Linking installer with light..."
& light.exe -ext WixUIExtension -ext WixUtilExtension -out "installer/dist/oido-$version.msi" "installer/staging/oido.wixobj"
if ($LastExitCode -ne 0) { throw "light.exe failed with exit code $LastExitCode" }

# 6. Generate SHA256 sidecar file
Write-Host "Generating SHA256 checksum..."
$msiPath = "installer/dist/oido-$version.msi"
$hash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLower()
"$hash  oido-$version.msi" | Out-File -FilePath "installer/dist/oido-$version.msi.sha256" -Encoding ascii

Write-Host "MSI Installer successfully built at installer/dist/oido-$version.msi"
Pop-Location
