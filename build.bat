@echo off

REM Create output directories
mkdir dist\linux_x86 2>nul
mkdir dist\linux_x86_64 2>nul
mkdir dist\windows 2>nul

set PKG_CONFIG_ALLOW_CROSS=1

REM Build for each target
echo Building for i686-unknown-linux-gnu...
cargo build --release --target i686-unknown-linux-gnu

echo Building for x86_64-unknown-linux-gnu...
cargo build --release --target x86_64-unknown-linux-gnu

echo Building for x86_64-pc-windows-gnu...
cargo build --release --target x86_64-pc-windows-gnu

REM Copy binaries to respective folders
copy target\i686-unknown-linux-gnu\release\IrisVA dist\linux_x86\
copy target\x86_64-unknown-linux-gnu\release\IrisVA dist\linux_x86_64\
copy target\x86_64-pc-windows-gnu\release\IrisVA.exe dist\windows\

REM Rename binaries
ren dist\linux_x86_64\IrisVA IrisVAx86_64
ren dist\linux_x86\IrisVA IrisVAx86

echo Build complete! Binaries are in the dist\ folder.
pause
