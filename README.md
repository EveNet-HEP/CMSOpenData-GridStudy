# CMSOpenData_GridStudy
This repository contains the code and documentation for the CMS Open Data Grid Study,
which is an analysis of the performance of the CMS Open Data on a grid computing infrastructure.
The study includes a detailed analysis of the data processing and storage requirements.

## Installation
```aiignore
git clone https://github.com/EveNet-HEP/CMSOpenData_GridStudy.git
cd CMSOpenData_GridStudy
```
To run the code, we requires the following dependencies:
- Python 3.12 or higher
- EveNet-Lite
```bash
conda create --prefix [path] python=3.12
conda activate [path]
```
### Option 1. Use our pip release
```
pip3 install evenet-lite 
```

### Option 2. Or install from source
```
git clone --recursive https://github.com/EveNet-HEP/EveNet-Lite.git
# Every time you open a new terminal, run this command to add the source code to your PYTHONPATH
cd EveNet-Lite; export PYTHONPATH=$(pwd):$PYTHONPATH # If you want to use the source code directly
```
