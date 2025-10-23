@echo off
echo Setting up Rust environment...
set PATH=%PATH%;%USERPROFILE%\.cargo\bin
cd /d "d:\OMS Chat Bot\qdrant"
echo Building Qdrant...
cargo build --release
echo Qdrant build complete!
pause