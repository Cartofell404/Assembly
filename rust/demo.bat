@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  End-to-end demo of the Assembly CLI on Windows, with no models to download.
rem
rem  The Windows counterpart of demo.sh. It creates GGUF-shaped files (a valid
rem  magic/version/count header, zero tensors, no metadata), writes a config
rem  that points at them, then runs the real binary through
rem  validate -^> build -^> info -^> info --verify.
rem
rem  These are not usable models. Nothing infers from them; they exist so the
rem  format, the pre-flight checks and the writer can be exercised without
rem  pulling gigabytes off the network.
rem
rem  Usage:
rem    demo.bat                   write the demo into rust\target\demo
rem    demo.bat --out <directory> write it somewhere else
rem ===========================================================================

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "OUT=%HERE%\target\demo"
set "BIN=%HERE%\target\debug\assembly.exe"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--out" goto arg_out
if /i "%~1"=="-out"  goto arg_out
call :err "Unknown option: %~1"
exit /b 1

:arg_out
if "%~2"=="" (
    call :err "--out needs a directory"
    exit /b 1
)
set "OUT=%~f2"
shift
shift
goto parse_args

:args_done

echo(==^> build
rem Build from the script's own directory, so the demo works whether it is run
rem from rust\ or from the repository root.
pushd "%HERE%"
cargo build -p assembly --quiet
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    call :err "cargo build failed"
    exit /b 1
)

if exist "%OUT%" rd /s /q "%OUT%"
md "%OUT%" >nul 2>&1

rem --- fake GGUF files -------------------------------------------------------

rem A GGUF with a header the validator accepts and no content worth loading:
rem 4 magic + 4 version + 8 tensor_count + 8 metadata_count = 24-byte header,
rem then zero padding. Version 3, zero tensors, zero metadata.
rem
rem certutil turns a file of hex bytes into the binary, so the header and a
rem 1 KB block of zeros are written without needing a second scripting runtime.
rem Every larger block is that 1 KB block doubled with `copy /b`.

set "WORK=%TEMP%\assembly-demo-%RANDOM%"
set "ZDIR=%WORK%\z"
md "%ZDIR%" >nul 2>&1

setlocal EnableDelayedExpansion
set "ZEROES="
for /l %%i in (1,1,512) do set "ZEROES=!ZEROES!00 "
> "%WORK%\zeroes.txt" echo(!ZEROES!
> "%WORK%\header.txt" echo(47 47 55 46 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
certutil -f -decodehex "%WORK%\zeroes.txt" "%ZDIR%\z1k.bin"   >nul 2>nul
set "DECODE_RC=!ERRORLEVEL!"
certutil -f -decodehex "%WORK%\header.txt" "%ZDIR%\header.bin" >nul 2>nul
if not "!DECODE_RC!"=="0" set "DECODE_RC=!ERRORLEVEL!"
endlocal & set "DECODE_RC=%DECODE_RC%"

if not "%DECODE_RC%"=="0" (
    call :err "certutil could not write the demo GGUF headers"
    rd /s /q "%WORK%" 2>nul
    exit /b 1
)

call :double z1k.bin   z2k.bin
call :double z2k.bin   z4k.bin
call :double z4k.bin   z8k.bin
call :double z8k.bin   z16k.bin
call :double z16k.bin  z32k.bin
call :double z32k.bin  z64k.bin
call :double z64k.bin  z128k.bin
call :double z128k.bin z256k.bin
call :double z256k.bin z512k.bin

pushd "%ZDIR%"
copy /b /y header.bin+z64k.bin            "%OUT%\route.gguf"   >nul
copy /b /y header.bin+z512k.bin           "%OUT%\general.gguf" >nul
copy /b /y header.bin+z512k.bin+z512k.bin "%OUT%\code.gguf"    >nul
copy /b /y header.bin+z128k.bin           "%OUT%\small.gguf"   >nul
popd

rem --- config ----------------------------------------------------------------

rem The Windows separators in the paths are doubled so they survive the trip
rem through a JSON string, where a lone backslash would be an escape.
set "OUTJ=%OUT:\=\\%"
set "CFG=%OUT%\config.json"
set "ARCHIVE=%OUT%\demo.asmb"

rem No BOM: the config reader parses the file as UTF-8 and a leading byte-order
rem mark would be a syntax error to the JSON parser. `echo` writes plain ASCII.
> "%CFG%" echo({
>>"%CFG%" echo(    "assembly_name": "Demo",
>>"%CFG%" echo(    "persona": "Be brief, then stop.",
>>"%CFG%" echo(    "embedding": {
>>"%CFG%" echo(        "name": "route",
>>"%CFG%" echo(        "path": "%OUTJ%\\route.gguf",
>>"%CFG%" echo(        "confidence_threshold": 0.6,
>>"%CFG%" echo(        "fallback_model": "general"
>>"%CFG%" echo(    },
>>"%CFG%" echo(    "models": [
>>"%CFG%" echo(        {
>>"%CFG%" echo(            "name": "general",
>>"%CFG%" echo(            "path": "%OUTJ%\\general.gguf",
>>"%CFG%" echo(            "anchors": ["how are you", "what is the weather"]
>>"%CFG%" echo(        },
>>"%CFG%" echo(        {
>>"%CFG%" echo(            "name": "code",
>>"%CFG%" echo(            "path": "%OUTJ%\\code.gguf",
>>"%CFG%" echo(            "anchors": ["write a function", "fix this bug"]
>>"%CFG%" echo(        },
>>"%CFG%" echo(        {
>>"%CFG%" echo(            "name": "small",
>>"%CFG%" echo(            "path": "%OUTJ%\\small.gguf",
>>"%CFG%" echo(            "anchors": ["hi", "thanks", "ok"]
>>"%CFG%" echo(        }
>>"%CFG%" echo(    ]
>>"%CFG%" echo(})

rem --- run it ----------------------------------------------------------------

echo(
echo(==^> validate
"%BIN%" validate "%CFG%"

echo(
echo(==^> build
"%BIN%" build "%CFG%" -o "%ARCHIVE%"

echo(
echo(==^> info
"%BIN%" info "%ARCHIVE%"

echo(
echo(==^> info --verify
"%BIN%" info "%ARCHIVE%" --verify

echo(
echo(archive: %ARCHIVE%

rd /s /q "%WORK%" 2>nul
exit /b 0

rem --- helpers ---------------------------------------------------------------

:double
pushd "%ZDIR%"
copy /b /y %~1+%~1 %~2 >nul
popd
exit /b 0

:err
echo([x] %~1
exit /b 0
