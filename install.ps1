# Assembly installer for Windows (PowerShell)
$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RustDir = Join-Path $RepoDir "rust"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  Write-Error "cargo not found. Install Rust via https://rustup.rs"
  exit 1
}

Write-Host "[+] Building and installing (first build takes 5-10 minutes)..." -ForegroundColor Green
Push-Location $RustDir
cargo install --path crates\cli --locked
Pop-Location

$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ($env:PATH -notlike "*$cargoBin*") {
  Write-Host "[!] $cargoBin not in PATH. Add it to User PATH if needed." -ForegroundColor Yellow
}

$ver = assembly --version 2>$null
if ($ver) { Write-Host "[+] Installed: $ver" -ForegroundColor Green }
else { Write-Host "[+] Installed successfully" -ForegroundColor Green }
Write-Host "[+] Binary: $(Get-Command assembly -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)" -ForegroundColor Green
