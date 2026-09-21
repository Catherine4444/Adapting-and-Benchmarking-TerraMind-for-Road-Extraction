import argparse
import subprocess
from pathlib import Path
import yaml
import shutil
import os 
import wandb
from datetime import datetime

BASE_DIR = Path("/teamspace/studios/this_studio/InstaRoadPrototype/src/finetuning_tm")


def consolidate_artefacts(root_dir, name, seed, rank):
    
    # 1. Top-level config artifacts written directly to BASE_DIR by terratorch/lightning
    config_dest_dir = root_dir / "config"
    config_dest_dir.mkdir(parents=True, exist_ok=True)   # <-- add this line

    for fname in ("config.yaml", "config_deploy.yaml", f"lora_linear_{name}_modules.yml"):
        src = BASE_DIR / fname
        if src.exists():
            shutil.move(str(src), str(config_dest_dir / fname))

    # 2. Experiment logs (timestamped subfolders)
    logs_src_dir = BASE_DIR / "experiment_logs"
    if logs_src_dir.exists():
        logs_dest_dir = root_dir / "experiment_logs"
        logs_dest_dir.mkdir(parents=True, exist_ok=True)
        for f in logs_src_dir.iterdir():
            shutil.move(str(f), str(logs_dest_dir / f.name))

    # 3. wandb logs (debug files + run folders)
    wandb_src_dir = BASE_DIR / "wandb"
    if wandb_src_dir.exists():
        wandb_dest_dir = root_dir / "wandb"
        wandb_dest_dir.mkdir(parents=True, exist_ok=True)

        for p in wandb_src_dir.glob("*.log"): 
            shutil.move(str(p), str(wandb_dest_dir / p.name))

        for f in wandb_src_dir.iterdir():
            if f.is_symlink():
                f.unlink()          # e.g. "latest-run" — don't move symlinks, just drop them
            elif f.name.startswith("run-"):
                shutil.move(str(f), str(wandb_dest_dir / f.name))
    return root_dir

def setup_data_cfg(data_cfg_path: Path, dataset_dir: str = "/teamspace/studios/this_studio/ROSADataset", num_workers: int = 6):
    with open(data_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["init_args"]["dataset_dir"] = dataset_dir
    cfg["init_args"]["num_workers"] = num_workers
    with open(data_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def setup_wandb_project(cfg_path: Path, wandb_project_name: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["trainer"]["logger"]["init_args"]["project"] = wandb_project_name
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def login_wandb():
    subprocess.run("wandb login --relogin ${WANDB_API_KEY}", cwd=BASE_DIR, check=True, shell=True)
    

def setup_rank_cfg(cfg_path: Path, r: int):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["init_args"]["model_args"]["peft_config"]["peft_config_kwargs"]["r"] = r
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def tune_lora_rank(name, seeds, ranks, max_epochs, precision, smoke_test):
    data_cgf_dir = BASE_DIR / "config/rosa_data"
    cgf_dir = BASE_DIR / f"config/peft"

    data_cfg_name = "base_data.yml"
    cfg_name = f"lora_linear_{name}_modules.yml"
    wandb_project_name = f"lora_linear_{name}_tune_rank_lightningai_runs_rosa"

    dest_cgf_dir = BASE_DIR / f"config/lightning_ai_copies_{name}" 
    dest_cgf_dir.mkdir(parents=True, exist_ok=True)  

    shutil.copy2(cgf_dir / cfg_name, dest_cgf_dir / cfg_name)
    shutil.copy2(data_cgf_dir / data_cfg_name, dest_cgf_dir / data_cfg_name)

    data_cfg_path = dest_cgf_dir / data_cfg_name
    cfg_path = dest_cgf_dir / cfg_name

    setup_data_cfg(data_cfg_path)
    setup_wandb_project(cfg_path, wandb_project_name)

    for seed in seeds:
        for rank in ranks:
            setup_rank_cfg(cfg_path, rank)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root_dir = BASE_DIR / f"artefacts/lora_{name}_seed{seed}_rank{rank}_{timestamp}"
            root_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "terratorch", "fit",
                "--config", str(cfg_path),
                "--trainer.max_epochs", str(max_epochs),
                "--seed_everything", str(seed),
                "--trainer.default_root_dir", str(root_dir),
                "--trainer.precision", precision,
                "--data", str(data_cfg_path),
                "--custom_modules_path", str(BASE_DIR),
            ]
            if smoke_test:
                cmd += [
                    "--trainer.limit_train_batches", "2",
                    "--trainer.limit_val_batches", "2",
                ]
            subprocess.run(cmd, cwd=BASE_DIR, check=True)

            ckpt_dir = BASE_DIR / "checkpoints" / f"lora_linear_{name}_modules"

            if ckpt_dir.exists():
                shutil.move(str(ckpt_dir), str(root_dir))

            consolidate_artefacts( root_dir, name, seed, rank)

    # Clean setup 
    config_src_dir = BASE_DIR / "artefacts" / "config" / f"lightning_ai_copies_{name}"  # note: also needs `name`, see below
    if config_src_dir.exists():
        config_dest_dir = BASE_DIR / "lai_copy_config"
        config_dest_dir.mkdir(parents=True, exist_ok=True)
        for f in config_src_dir.glob("*.yml"):
            shutil.move(str(f), str(config_dest_dir / f.name))

        for f in config_src_dir.glob("*.yaml"):
            shutil.move(str(f), str(config_dest_dir / f.name))

if __name__ == "__main__":
    # wandb_api_key = os.getenv("wandb")
    # if wandb_api_key:
    #     wandb.login(key=wandb_api_key)
    # else:
    #     raise ValueError("WANDB_API_KEY secret was not found in environment variables.")
    login_wandb()
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="attn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--smoke_test", action="store_true",
                         help="Caps batches for a fast CPU sanity check")
    args = parser.parse_args()
    tune_lora_rank(args.name, args.seeds, args.ranks, args.max_epochs, args.precision, args.smoke_test)



