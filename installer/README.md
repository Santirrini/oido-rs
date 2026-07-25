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

Oido verifica los updates descargados con criptografía de clave pública (Ed25519 via `minisign`). El binario embebe `installer/updater-pubkey.txt` en tiempo de compilación (`include_str!` en `crates/oido-updater/src/verify.rs`). El flujo es:

1. **Defensa en profundidad**:
   - SHA-256 sidecar (`*.msi.sha256`) verifica integridad bit-a-bit.
   - Firma Ed25519 sidecar (`*.msi.minisig`) verifica **autenticidad**: que el MSI venga realmente del mantenedor y no de un MITM.
2. **Defense in depth en orden estricto**: SHA-256 primero (rápido, descarta corrupciones), luego Ed25519 (autentica origen). Si cualquiera falla, el update se rechaza con `UpdateError::ChecksumMismatch` o `UpdateError::SignatureInvalid`.
3. **El binario rechaza** releases sin `.minisig` (configurado en `WindowsMsiBackend::requires_signature() = true`).

### Cómo generar las llaves

```bash
# 1. Generar el par (en una workstation segura, NO en CI runners compartidos).
minisign -G -p updater-pubkey.txt -s updater-privkey.key -W

# 2. Confirmar `installer/updater-pubkey.txt` (es público, va al repo).

# 3. La private key:
#    - Subirla como GitHub Secret `MINISIGN_PRIVATE_KEY` (Settings →
#      Secrets → Actions → New repository secret).
#    - NO commitearla al repo. La key actual que estaba en
#      `installer/updater-privkey.pem` fue removida por estar expuesta;
#      rotar la key pair antes del próximo release.
```

El workflow `.github/workflows/release.yml` lee el secret y firma el MSI en CI; `build-msi.ps1` también firma localmente si el secret y `minisign.exe` están disponibles.

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
