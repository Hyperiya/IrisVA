#!/bin/bash

# Create output directories
mkdir -p dist/{linux_x86,linux_x86_64,windows}

export PKG_CONFIG_ALLOW_CROSS=1

# Build for each target
echo "Building for i686-unknown-linux-gnu..."
cargo build --release --target i686-unknown-linux-gnu

echo "Building for x86_64-unknown-linux-gnu..."
cargo build --release --target x86_64-unknown-linux-gnu

echo "Building for x86_64-pc-windows-gnu..."
cargo build --release --target x86_64-pc-windows-gnu

# Copy binaries to respective folders
cp target/i686-unknown-linux-gnu/release/IrisVA dist/linux_x86/
cp target/x86_64-unknown-linux-gnu/release/IrisVA dist/linux_x86_64/
cp target/x86_64-pc-windows-gnu/release/IrisVA.exe dist/windows/

mv dist/linux_x86_64/IrisVA dist/linux_x86_64/IrisVAx86_64
mv dist/linux_x86/IrisVA dist/linux_x86/IrisVAx86
mv dist/windows/IrisVA.exe dist/windows/IrisVA.exe

echo "Build complete! Binaries are in the dist/ folder."
