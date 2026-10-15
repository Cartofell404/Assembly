@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  Assembly installer for 64-bit Windows.
rem
rem  The Windows counterpart of install.sh: it checks the platform, makes sure
rem  Rust and the tools llama.cpp needs are present, then builds and installs
rem  the `assembly` command with cargo.
rem
rem  Building from source compiles llama.cpp the first time, which needs three
rem  separate things on Windows:
rem
rem    * a C++ toolchain -- the default Rust target on Windows is msvc, so this
rem      means the MSVC Build Tools (link.exe and the Windows SDK);
rem    * CMake, which llama.cpp's build uses;
rem    * LLVM, because bindgen runs clang (via libclang.dll) to generate the
rem      FFI bindings for llama.cpp.
rem
rem  Missing tools are reported and, with --install-deps, installed through
rem  winget. Installing the MSVC Build Tools is several gigabytes and needs
rem  administrator rights, so it is never done silently.
rem
rem  Usage:
rem    install.bat                 check the toolchain, then build and install
rem    install.bat --install-deps  also install missing prerequisites with winget
rem    install.bat --skip-build    environment checks only
rem ===========================================================================

set "INSTALL_DEPS="
set "SKIP_BUILD="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--install-deps" goto arg_install_deps
if /i "%~1"=="-installdeps"   goto arg_install_deps
if /i "%~1"=="--skip-build"   goto arg_skip_build
if /i "%~1"=="-skipbuild"     goto arg_skip_build
call :err "Unknown option: %~1"
echo(
call :usage
exit /b 1

:arg_install_deps
set "INSTALL_DEPS=1"
shift
goto parse_args

:arg_skip_build
set "SKIP_BUILD=1"
shift
goto parse_args

:args_done

rem --- environment -----------------------------------------------------------

set "PF86=%ProgramFiles(x86)%"

if defined CARGO_HOME (
    set "CARGO_BIN=%CARGO_HOME%\bin"
) else (
    set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
)

rem --- platform check --------------------------------------------------------

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
if /i "%ARCH%"=="AMD64" goto arch_ok
if /i "%ARCH%"=="ARM64" goto arch_ok
call :err "Assembly needs 64-bit Windows; this is a %PROCESSOR_ARCHITECTURE% process."
exit /b 1
:arch_ok

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "RUST_DIR=%REPO_DIR%\rust"

if not exist "%RUST_DIR%\Cargo.toml" (
    call :err "Cannot find rust\Cargo.toml under %REPO_DIR%. Run this from a full checkout."
    exit /b 1
)

echo(=== Assembly Installer ^(Windows^) ===
call :info "Repository: %REPO_DIR%"
call :info "Architecture: %ARCH%"
echo(

call :ensure_rust
if errorlevel 1 exit /b 1

call :ensure_build_deps
if errorlevel 1 exit /b 1

if defined SKIP_BUILD (
    call :info "--skip-build set; environment checks passed."
    exit /b 0
)

call :build_and_install
if errorlevel 1 exit /b 1

call :test_install

echo(
echo(=== Installation Complete ===
call :log "Run 'assembly --help' to get started"
call :log "Run uninstall.bat to remove Assembly"
exit /b 0

rem --- Rust ------------------------------------------------------------------

:ensure_rust
call :test_command cargo
if not errorlevel 1 goto rust_found
if exist "%CARGO_BIN%\cargo.exe" goto rust_local

call :log "Rust not found; installing it via rustup..."
set "RUSTUP_INIT=%TEMP%\rustup-init.exe"
rem The official rustup bootstrap. Fetched over HTTPS from the canonical host;
rem no inline execution of remote script content.
curl.exe -sS -L -o "%RUSTUP_INIT%" https://win.rustup.rs/x86_64
if errorlevel 1 (
    call :err "Could not download rustup-init. Is curl available on this system?"
    exit /b 1
)
"%RUSTUP_INIT%" -y --default-toolchain stable --profile default
if errorlevel 1 (
    call :err "rustup-init failed"
    exit /b 1
)
set "PATH=%CARGO_BIN%;%PATH%"
call :log "Rust installed."
exit /b 0

:rust_found
for /f "delims=" %%v in ('cargo --version') do set "RUSTVER=%%v"
call :log "Rust already installed: %RUSTVER%"
exit /b 0

:rust_local
call :log "Rust found at %CARGO_BIN%\cargo.exe"
set "PATH=%CARGO_BIN%;%PATH%"
exit /b 0

rem --- build prerequisites ---------------------------------------------------

:ensure_build_deps
set "MISSING="
call :test_msvc_toolchain
if errorlevel 1 set "MISSING=%MISSING% msvc"
call :test_command cmake
if errorlevel 1 set "MISSING=%MISSING% cmake"
call :test_libclang
if errorlevel 1 set "MISSING=%MISSING% llvm"

if not defined MISSING goto deps_present

call :warn "Missing build prerequisites:%MISSING%"
if defined INSTALL_DEPS goto deps_install

echo(
call :info "Run again with --install-deps to install these with winget, or install them by hand:"
echo(  MSVC Build Tools : winget install -e --id Microsoft.VisualStudio.2022.BuildTools
echo(                     ^(select the "Desktop development with C++" workload^)
echo(  CMake            : winget install -e --id Kitware.CMake
echo(  LLVM ^(libclang^)  : winget install -e --id LLVM.LLVM
exit /b 1

:deps_present
call :log "Build prerequisites present (MSVC, CMake, LLVM/libclang)."
rem Help bindgen find libclang when LLVM was installed but the variable was
rem never exported for this shell.
if defined LIBCLANG_PATH exit /b 0
if exist "C:\Program Files\LLVM\bin\libclang.dll" set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
exit /b 0

:deps_install
call :test_command winget
if errorlevel 1 (
    call :err "winget is not available; install the prerequisites above by hand."
    exit /b 1
)

echo %MISSING% | find /i "cmake" >nul 2>&1
if not errorlevel 1 (
    call :install_winget_package Kitware.CMake
    if errorlevel 1 call :warn "CMake install failed"
)

echo %MISSING% | find /i "llvm" >nul 2>&1
if not errorlevel 1 (
    call :install_winget_package LLVM.LLVM
    if errorlevel 1 call :warn "LLVM install failed"
)

echo %MISSING% | find /i "msvc" >nul 2>&1
if not errorlevel 1 (
    call :warn "Installing the MSVC Build Tools is large and needs administrator rights."
    call :install_winget_package Microsoft.VisualStudio.2022.BuildTools "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    if errorlevel 1 call :warn "MSVC Build Tools install did not complete"
)

rem PATH changes made by installers are not visible to this process; refresh
rem from the machine and user scopes so the build can find the new tools.
set "MACHINE_PATH="
set "USER_PATH="
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul ^| find /i "REG_"') do set "MACHINE_PATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul ^| find /i "REG_"') do set "USER_PATH=%%B"
if defined MACHINE_PATH set "PATH=%MACHINE_PATH%;%USER_PATH%"

if defined LIBCLANG_PATH goto deps_recheck
if exist "C:\Program Files\LLVM\bin\libclang.dll" set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"

:deps_recheck
set "STILL="
call :test_msvc_toolchain
if errorlevel 1 set "STILL=%STILL% MSVC"
call :test_command cmake
if errorlevel 1 set "STILL=%STILL% CMake"
call :test_libclang
if errorlevel 1 set "STILL=%STILL% LLVM"

if not defined STILL exit /b 0
call :warn "Still missing after install:%STILL%. You may need a new shell ^(or a reboot^) for the changes to take effect."
exit /b 1

:test_msvc_toolchain
rem link.exe is what rustc's msvc target needs; the Windows SDK provides the
rem rest. vswhere reporting the VC tools component is the reliable check.
call :test_command link.exe
if not errorlevel 1 exit /b 0
if not defined PF86 exit /b 1
if not exist "%PF86%\Microsoft Visual Studio\Installer\vswhere.exe" exit /b 1
set "VSINSTALL="
"%PF86%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%TEMP%\assembly-vswhere.txt" 2>nul
set /p VSINSTALL=<"%TEMP%\assembly-vswhere.txt"
if defined VSINSTALL exit /b 0
exit /b 1

:test_libclang
if defined LIBCLANG_PATH if exist "%LIBCLANG_PATH%\libclang.dll" exit /b 0
call :test_command clang.exe
if not errorlevel 1 exit /b 0
if exist "C:\Program Files\LLVM\bin\libclang.dll" exit /b 0
if exist "C:\Program Files (x86)\LLVM\bin\libclang.dll" exit /b 0
exit /b 1

:install_winget_package
call :test_command winget
if errorlevel 1 exit /b 1
call :info "winget install %~1"
if "%~2"=="" (
    winget install --id %~1 -e --accept-source-agreements --accept-package-agreements
) else (
    winget install --id %~1 -e --accept-source-agreements --accept-package-agreements --override "%~2"
)
exit /b %ERRORLEVEL%

rem --- build and install -----------------------------------------------------

:build_and_install
call :log "Building and installing Assembly (first build compiles llama.cpp; 5-15 minutes)..."
pushd "%RUST_DIR%"
cargo install --path crates/cli --locked --force
set "BUILD_RC=%ERRORLEVEL%"
popd
if not "%BUILD_RC%"=="0" (
    call :err "cargo install failed"
    exit /b 1
)
call :log "Build and install finished."
exit /b 0

:test_install
if not exist "%CARGO_BIN%\assembly.exe" (
    call :warn "Could not find assembly.exe in %CARGO_BIN%."
    exit /b 0
)
set "ASMVER="
"%CARGO_BIN%\assembly.exe" --version >"%TEMP%\assembly-version.txt" 2>nul
set /p ASMVER=<"%TEMP%\assembly-version.txt"
call :log "Assembly %ASMVER% is installed and ready"
call :info "Location: %CARGO_BIN%\assembly.exe"
echo ;%PATH%; | find /i ";%CARGO_BIN%;" >nul 2>&1
if not errorlevel 1 exit /b 0
call :warn "%CARGO_BIN% is not on your PATH for this session."
call :info "Open a new terminal, or run:"
echo(  set "PATH=%CARGO_BIN%;%%PATH%%"
exit /b 0

rem --- helpers ---------------------------------------------------------------

:usage
echo(Usage: install.bat [options]
echo(
echo(  --install-deps  Install missing prerequisites through winget. Without
echo(                  this flag the script reports what is missing and stops.
echo(  --skip-build    Check the environment only; do not build or install
echo(                  anything.
exit /b 0

:test_command
where %~1 >nul 2>&1
exit /b %ERRORLEVEL%

:log
echo([+] %~1
exit /b 0

:info
echo([i] %~1
exit /b 0

:warn
echo([!] %~1
exit /b 0

:err
echo([x] %~1
exit /b 0
