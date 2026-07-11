use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

pub const REPO: &str = "Santirrini/oido-rs";
pub const BIN_NAME: &str = "oido";
#[allow(dead_code)]
pub const PUBLIC_KEY: &str = include_str!("../../../installer/updater-pubkey.txt");

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Status {
    UpToDate,
    DownloadedAndInstalling { version: String },
}

pub fn check_and_apply() -> Result<Status> {
    let (owner, repo) = match REPO.split_once('/') {
        Some((o, r)) => (o, r),
        None => return Err(anyhow!("Invalid REPO format: {}", REPO)),
    };

    let updater = self_update::backends::github::Update::configure()
        .repo_owner(owner)
        .repo_name(repo)
        .bin_name(BIN_NAME)
        .current_version(env!("CARGO_PKG_VERSION"))
        .build()
        .context("Failed to build self_update configuration")?;

    let latest_release = updater
        .get_latest_release()
        .context("Failed to fetch latest release from GitHub")?;

    let latest_version = latest_release.version.clone();
    let current_version = env!("CARGO_PKG_VERSION");

    let is_greater =
        self_update::version::bump_is_greater(current_version, &latest_version).unwrap_or(false);

    if !is_greater {
        return Ok(Status::UpToDate);
    }

    let msi_asset = latest_release
        .assets
        .iter()
        .find(|a| a.name.ends_with(".msi"))
        .ok_or_else(|| anyhow!("No .msi asset found in release v{}", latest_version))?;

    let sha256_asset = latest_release
        .assets
        .iter()
        .find(|a| a.name.ends_with(".msi.sha256"))
        .ok_or_else(|| {
            anyhow!(
                "No .msi.sha256 sidecar asset found in release v{}",
                latest_version
            )
        })?;

    let temp_dir = std::env::temp_dir();
    let msi_path = temp_dir.join(format!("oido-{}.msi", latest_version));
    let sha_path = temp_dir.join(format!("oido-{}.msi.sha256", latest_version));

    download_file(&msi_asset.download_url, &msi_path)
        .context("Failed to download installer .msi file")?;

    download_file(&sha256_asset.download_url, &sha_path)
        .context("Failed to download SHA256 checksum file")?;

    verify_sha256(&msi_path, &sha_path)
        .context("Installer SHA256 integrity verification failed")?;

    let msi_path_str = msi_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid path for msi_path"))?;

    Command::new("msiexec")
        .args(["/i", msi_path_str, "/qb", "/norestart"])
        .spawn()
        .context("Failed to spawn msiexec installer")?;

    Ok(Status::DownloadedAndInstalling {
        version: latest_version,
    })
}

pub fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("oido-updater/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("Failed to create reqwest client")?;

    let mut response = client.get(url).send().context("Failed to send request")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let mut file = File::create(dest).context("Failed to create destination file")?;

    response
        .copy_to(&mut file)
        .context("Failed to write downloaded data to file")?;

    Ok(())
}

pub fn verify_sha256(msi_path: &Path, sha_path: &Path) -> Result<()> {
    let mut sha_content = String::new();
    File::open(sha_path)
        .context("Failed to open SHA256 checksum file")?
        .read_to_string(&mut sha_content)
        .context("Failed to read SHA256 checksum file")?;

    let expected_hash = sha_content
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("SHA256 checksum file is empty or malformed"))?
        .to_lowercase();

    if expected_hash.len() != 64 || !expected_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "Invalid SHA256 hash in checksum file: '{}'",
            expected_hash
        ));
    }

    let mut msi_file = File::open(msi_path).context("Failed to open downloaded .msi file")?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = msi_file
            .read(&mut buffer)
            .context("Failed to read downloaded .msi file")?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let result = hasher.finalize();
    let computed_hash = hex::encode(result);

    if computed_hash.to_lowercase() != expected_hash {
        return Err(anyhow!(
            "SHA256 mismatch! Expected: {}, Computed: {}",
            expected_hash,
            computed_hash
        ));
    }

    Ok(())
}
