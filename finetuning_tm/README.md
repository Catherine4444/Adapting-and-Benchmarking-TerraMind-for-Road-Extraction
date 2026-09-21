# Adapting and Benchmarking TerraMind 
This folder contains the code for:
1. adapting the Geospation Foundation Model TerraMind through decoder selection and LoRA finetuning 
2. benchmarking TerraMind against the medium-sized task-specific models D-LinkNet and U-Net. 

## Project Abstract
Automating road extraction from open, 10 m-resolution remote sensing imagery is drastically more difficult than using high-resolution datasets (< 1𝑚), which often focuses on urban areas, because of the thin typology of roads that can reach sub-pixel width. This leaves rural and peri-urban areas underserved, yet these areas remain a vital part of tasks such as disaster management and environmental monitoring. The development Geospatial Foundation Models (GFMs) offers the opportunity to better leverage multi-modal imagery from the open-access Sentinel 1 and 2 missions with a high revisit frequency of 5-days. As a result, a dataset using Sentinel-1/2 imagery and spanning urban, peri-urban and rural areas was curated to investigate using the multi-modal GFM TerraMind for road extraction under five decoder architectures, LoRA configurations across rank and target model, as well as benchmarking against the task-specific baselines, U-Net and D-LinkNet. The Fully convolutional Network (FCN) decoder under full fine-tuning was the strongest TerraMind adaptation (IoU=0.415). The vanilla LoRA setup struggled to capture the high "intrinsic rank" of road extraction and could not match full fine-tuning. Finally, the fully fine-tuned U-Net outperforms every TerraMind configuration, with only a quarter of the parameters in comparison (IoU=0.486). This gap is attributed to U-Net’s multi-scale feature fusion that allows resolution of up to 20m to be preserved and leveraged in the decoder in contrast to the 160 m x 160 m resolution tokens inputted into the TerraMind encoder. The findings suggest that the adaptation approaches lack the resolution and representational capacity to match task-specific models on road extraction.

## Project Structure
The project code is organised as follows. 
```text
config/ YAML config files for TerraTorch CLI to tune and train TerraMind and the baseline models on the UCT HPC (and get model checkpoints). Additionally the benchmark config is for the `benchmarking` module used during test set evaluation
custom_modules/ the custom modules needed for the YAML configurations. 
hpc_scripts/ The bash and sbatch scripts to run the config files 
kaggle_prep/ The setup script and code for the kaggle environments. 
kaggle_prototypes/ The Juypiter notebooks used in prototyping and conducting preliminary experiments 
test_results.ipynb The notebook running th final test split results using previously obtained checkpoints. It uses the external `benchmarking` module, TerraTorch test, and TerraTorch Predict.
```
The project is mostly self contained in the folder. The only external module is the benchmarking module that is used in the test_result.ipynb. 

## Requirements
The project uses uv packet manager and to set up the relevant virtual environment, run the following:
```
pip install uv
uv pip install --system -e "InstaRoadPrototype[terra,unet, dlinknet, finetuning_tm]"
```



