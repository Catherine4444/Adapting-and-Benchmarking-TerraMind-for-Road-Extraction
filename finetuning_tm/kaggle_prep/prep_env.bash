
set -e
cd /kaggle/working

echo "Installing python dependencies..."

pip install uv
uv pip install --system -e "InstaRoadPrototype[terra,unet, dlinknet, finetuning_tm]"

echo "Environment ready! "