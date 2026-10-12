@echo off
REM Assembly - Complete Windows installer (batch version)
REM Installs Rust, builds and installs Assembly globally

setlocal enabledelayedexpansion

echo === Assembly Complete Installer ===
echo.

REM Check if running as admin (optional but helpful)
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [+] Running as administrator
) else (
    echo [!] Not running as administrator (some deps may require admin)
)

REM Get repo dir
set REPO_DIR=%~dp0
set RUST_DIR=%REPO_DIR%rust
set CARGO_HOME=%USERPROFILE%\.cargo
set CARGO_BIN=%CARGO_HOME%\bin

REM Install Rust if not present
where rustc >nul 2>&1
if %errorlevel% neq 0 (
    echo [+] Rust not found. Installing via rustup-init.exe...
    echo [i] Downloading rustup...
    curl -L -o "%TEMP%\rustup-init.exe" https://win.rustup.rs/x86_64
    if %errorlevel% neq 0 (
        echo [x] Failed to download rustup. Please install from https://rustup.rs
        pause
        exit /b 1
    )
    echo [i] Running rustup installer...
    "%TEMP%\rustup-init.exe" -y --default-toolchain stable
    del "%TEMP%\rustup-init.exe" 2>nul
) else (
    echo [+] Rust already installed: 
    rustc --version
)

REM Ensure cargo in PATH for this session
set PATH=%CARGO_BIN%;%PATH%

REM Install build dependencies (Windows)
echo [+] Checking build tools...
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] CMake not found. Attempting to install via winget/choco...
    where winget >nul 2>&1
    if %errorlevel% == 0 (
        winget install Kitware.CMake -e --accept-source-agreements --accept-package-agreements 2>nul || echo [!] winget install failed
    )
    where choco >nul 2>&1
    if %errorlevel% == 0 (
        choco install cmake -y 2>nul || echo [!] choco install failed
    )
)

where clang >nul 2>&1
if %errorlevel% neq 0 (
    where winget >nul 2>&1
    if %errorlevel% == 0 (
        winget install LLVM.LLVM -e --accept-source-agreements --accept-package-agreements 2>nul || echo [!] winget install failed
    )
    where choco >nul 2>&1
    if %errorlevel% == 0 (
        choco install llvm -y 2>nul || echo [!] choco install failed
    )
)

echo [+] Building and installing Assembly (this may take 5-15 minutes)...
cd /d "%RUST_DIR%"
cargo install --path crates\cli --locked --force

if %errorlevel% neq 0 (
    echo [x] Installation failed
    pause
    exit /b 1
)

echo [+] Assembly installed successfully
where assembly >nul 2>&1 && assembly --version

REM Create uninstaller
echo [+] Creating uninstaller...
(
echo @echo off
echo echo Uninstalling Assembly...
echo cargo uninstall assembly 2^>nul
echo echo Done
echo pause
) > "%REPO_DIR%\uninstall.bat"

echo.
echo === Installation Complete ===
echo Run 'assembly --help' to get started
echo Run 'uninstall.bat' to remove Assembly
pause
