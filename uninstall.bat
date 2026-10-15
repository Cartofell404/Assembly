@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  Remove the Assembly command installed by install.bat.
rem
rem  Mirrors uninstall.sh: `cargo uninstall assembly`, then clears any stray
rem  copy the installer may have linked into the user's local bin. It does not
rem  touch the Rust toolchain or the build prerequisites.
rem ===========================================================================

if defined CARGO_HOME (
    set "CARGO_BIN=%CARGO_HOME%\bin"
) else (
    set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
)
set "LOCAL_BIN=%USERPROFILE%\.local\bin"

call :log "Uninstalling Assembly..."

where cargo >nul 2>&1
if errorlevel 1 goto no_cargo

cargo uninstall assembly 2>nul
if errorlevel 1 call :warn "cargo uninstall failed (Assembly may not have been installed with cargo)"
goto remove_copy

:no_cargo
call :warn "cargo is not on PATH; removing the binary directly instead."
del /q "%CARGO_BIN%\assembly.exe" 2>nul

:remove_copy
rem install.sh links into ~/.local/bin; clear the same place on Windows if a
rem copy was dropped there by hand.
del /q "%LOCAL_BIN%\assembly.exe" 2>nul

call :log "Assembly uninstalled successfully"
exit /b 0

:log
echo([-] %~1
exit /b 0

:warn
echo([!] %~1
exit /b 0
