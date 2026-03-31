# CMSOpenData GridStudy
This repository contains the code and documentation for the CMS Open Data Grid Study,
which is an analysis of the performance of the CMS Open Data on a grid computing infrastructure.
The study includes a detailed analysis of the data processing and storage requirements.

## Installation
```aiignore
git clone https://github.com/EveNet-HEP/CMSOpenData-GridStudy.git
cd CMSOpenData-GridStudy
```
To run the code, we requires the following dependencies:
- Python 3.12 or higher
- EveNet-Lite
```bash
conda create --prefix [path] python=3.12
conda activate [path]
pip3 install -r requirements.txt
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
### To Enable wandb logging [Optional]
```aiignore
export WANDB_API_KEY=[your_wandb_api_key]
```

### Download Pre-trained Models
To download the pre-trained models from [Hugging Face Hub](https://huggingface.co/Avencast/EveNet), you can use the following command (`local-dir` could be replaced with any path you want)
```bash
hf download Avencast/EveNet --local-dir pretrain-weights
# nominal ckpt: checkpoints.20M.a4.last.ckpt
# SSL ckpt: SSL.20M.last.ckpt
```


## Data Inputs
### From CMS Open Data [Optional]
The CMS Open Data used in this study is available at [CMS Open Data Portal](https://opendata.cern.ch/). 
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
```
To run the ntuple production
```bash
# If you have multiple CPU cores, you can specify the number of workers to speed up the processing
python3 process_data.py Farm/output_list.json --workers [nCPU] --outdir [output/ntupledir]
```

### From Hugging Face Datasets [Recommended]
We also provide the processed datasets on [Hugging Face Hub](https://huggingface.co/datasets/Avencast/EveNet-GridStudy-CMSOpenData), which can be easily loaded with the following command:
(`local-dir` could be replaced with any path you want to store the dataset)

#### 🔹 Download full dataset
```bash
hf download Avencast/EveNet-GridStudy-CMSOpenData \
  --repo-type dataset \
  --local-dir database
````
#### 🔹 Download background samples only
```bash
hf download Avencast/EveNet-GridStudy-CMSOpenData \
  DYBJets_pt100to200 DYBJets_pt200toInf \
  ggHtautau VBFHtautau \
  SingleTop_s_channel_lepon \
  SingleTop_t_channel_top \
  SingleTop_t_channel_antitop \
  tt1l tW_top tW_antitop \
  --repo-type dataset \
  --local-dir database_bkg \
  --resume-download
```
#### 🔹 Download a single signal point (quick test)
```bash
hf download Avencast/EveNet-GridStudy-CMSOpenData \
  MX-300_MY-80 \
  --repo-type dataset \
  --local-dir database_test \
  --resume-download
```

## Machine Learning 
### EveNet
`--stage` configures the training stage, which can be set to `train`, `predict`, `evaluate`. It could also be sequentially run with `--stage train predict evalute`. 
```aiignore
# Nominal pretrain
python3 train_pc_mva.py --base_dir database --yaml_path config/sample_bbWW.yaml --mX 500 --mY 90 --out_dir result --learning_rate 0.0003  --pretrain pretrain-weights/checkpoints.20M.a4.last.ckpt --stage train predict evaluate  --use_adapter --wandb_dir /tmp --batch_size 4096 --gamma 0.0 --epochs 25
python3 train_pc_mva.py --base_dir database --yaml_path config/sample_bbWW.yaml --mX 500 --mY 90 --out_dir result --learning_rate 0.0003  --pretrain pretrain-weights/SSL.20M.last.ckpt --stage train predict evaluate  --use_adapter --wandb_dir /tmp --batch_size 4096 --gamma 0.0 --epochs 25
python3 train_pc_mva.py --base_dir database --yaml_path config/sample_bbWW.yaml --mX 500 --mY 90 --out_dir result --learning_rate 0.0003   --stage train predict evaluate --wandb_dir /tmp --batch_size 4096 --gamma 0.0 --epochs 25
```
Output will be
```aiignore
result/[mva_method]/individual/MX-[mX]_MY-[mY]/
├── checkpoints/
│   ├── model_epoch_0.pt
├── training_log.json # Training parameters and metrics logged during training
├── prediction_xxx.pnz # Prediction results for the test set
├── eval_metrics_xxx.json # Evaluation metrics such as AUC, accuracy, etc.
```
### XGBoost/TabPFN
```aiignore
python3 train_tabular_mva.py --base_dir database --yaml_path config/sample_bbWW.yaml --features_yaml config/feature_bbWW.yaml --out_dir [out_dir] --model [xgb/tabpfn] --mX [mx] --mY [my] --stage train evaluate predict
```
Output will be similar as EveNet ones.
## Grid Script Generation [Optional]
To generate the grid scripts for the ntuple production, you can use the following command:
```bash
python3 Make_script.py --farm_dir Farm --json_file Farm/output_list.json --data_dir database --out_dir result --pretrain-weight pretrain-weights/checkpoints.20M.a4.last.ckpt
```
This will create the scripts to run full grid scan. i.e. `Farm/run_[method]_[stage].sh`. It will consist all the needed command, you can run the target stage/method then, but suggest to use parallel running
for full grid study.



