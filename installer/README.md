# Oido Windows Installer (MSI) & Update System

This directory contains the WiX Toolset configuration and the orchestration script to package Oido as a Windows Installer (.msi) file.

## Prerequisites

To build the MSI package locally, you must install the following tools and ensure they are available in your system `PATH`:
1. **WiX Toolset v3.14** (or later v3.x release).
2. **Rust & Cargo** (installed via rustup).
3. **PowerShell** (built-in on Windows).

## WiX Command Line Tools
The build script relies on two key tools from the WiX Toolset:
- `candle.exe`: The WiX compiler that compiles the `.wxs` source file into a `.wixobj` intermediate file.
- `light.exe`: The WiX linker that links the `.wixobj` file into the final `.msi` package. The linker is configured to load the standard extensions `WixUIExtension` and `WixUtilExtension` for standard minimal UI dialogues.

## UpgradeCode (CRITICAL)
The installer defines a fixed `UpgradeCode` in `oido.wxs`:
```xml
UpgradeCode="B9A8A529-65A0-449D-BBF9-2A835D8B41D8"
```
> [!IMPORTANT]
> **NEVER MODIFY THE `UpgradeCode`.**
> Changing this GUID will break the installer upgrade sequence. Windows Installer uses the `UpgradeCode` to detect prior installations, perform silent uninstalls of previous versions, and migrate user settings during updates.

## Auto-Update Verification Keys

Oido verifies the downloaded updater MSI files using public-key cryptography (Ed25519).

### How to Generate Keys
To generate a new key pair for release signing:
1. Use `minisign` or any compatible Ed25519 key generation tool.
2. Store the private key in a secure location (e.g., as a GitHub repository secret).
3. Commit the public key to `installer/updater-pubkey.txt`. It will be embedded into the application executable at compilation time (`crates/oido/src/updater.rs` reads this file using `include_str!`).

## How to Build the Installer
Open a PowerShell terminal at the root of the project and execute:
```powershell
.\installer\build-msi.ps1
```
The script will:
1. Compile the Oido executable in release mode with the `updater` feature active.
2. Copy the binary to a temporary `staging/` directory.
3. Call `candle.exe` and `light.exe` to generate the MSI installer package.
4. Compute the SHA256 checksum of the installer and save it in a `.sha256` sidecar file.
5. Place the final output MSI and checksum file in the `installer/dist/` folder.
