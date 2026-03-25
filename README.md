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

## Data Information
The CMS Open Data used in this study is available at [CMS Open Data Portal](https://opendata.cern.ch/record/10000). 
The data includes a variety of datasets. In this study, we focus on the $X\rightarrow YH\rightarrow b\bar{b}WW$ channel, 
where we stores all the singal and background samples in the `config/samples.yaml` file.
```yaml
signal:
  type: signal
  wildcard: "NMSSM_XToYHTo2B2WTo2B2Q1L1Nu*"
  mx: [0, 1000]
  my: [0, 1000]

background:
  tt1l:
    wildcard: "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
    xsec: 365.35 # Cross section in pb
    max_events: 500000 # Maximum number of events to process, set to -1 for all events
    nEvent: 144722000 # Obtain from CMS Open Data Portal
    name: ttbar # Name of the background process
```
To fetch the input information
```bash
python3 resolve_sample.py --yaml config/sample_bbWW.yaml --output Farm/output_list.json
111
