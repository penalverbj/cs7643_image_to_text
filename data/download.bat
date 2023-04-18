@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set "ROOT_DIR=%%i"
cd %ROOT_DIR%\data

REM https://cocodataset.org/#download
REM https://github.com/cocodataset/cocoapi/issues/368
if not exist ".\\coco" (
    mkdir coco
    cd coco

    powershell -command "Invoke-WebRequest http://images.cocodataset.org/zips/train2017.zip -OutFile train2017.zip"
    powershell -command "Expand-Archive train2017.zip -DestinationPath ./ -Force"
    powershell -command "Remove-Item train2017.zip"
    powershell -command "Invoke-WebRequest http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip"
    powershell -command "Expand-Archive val2017.zip -DestinationPath ./ -Force"
    powershell -command "Remove-Item val2017.zip"
    powershell -command "Invoke-WebRequest http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile annotations_trainval2017.zip"
    powershell -command "Expand-Archive annotations_trainval2017.zip -DestinationPath ./ -Force"
    powershell -command "Remove-Item annotations_trainval2017.zip"

    cd ..
)
