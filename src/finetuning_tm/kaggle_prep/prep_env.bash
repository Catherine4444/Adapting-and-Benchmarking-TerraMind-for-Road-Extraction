
set -e
cd /kaggle/working

echo "Installing python dependencies..."

pip install uv
uv pip install --system -e "Adapting-and-Benchmarking-TerraMind-for-Road-Extraction[terra,unet, dlinknet, finetuning_tm]"

echo "Environment ready! "