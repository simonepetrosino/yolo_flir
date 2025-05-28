import wandb
from rfdetr import RFDETRBase

def main():
    wandb.init(
        project="rf-detr-flir",
        name="train_rf_detr_with_wandb",
        config={
            "dataset_dir": "C:/Users/simo5/Desktop/yoloprovalocale/flirdatasetcoco",
            "output_dir": "C:/Users/simo5/Desktop/yoloprovalocale/rf_detr_model",
            "batch_size": 4,
            "num_epochs": 10,
            "grad_accum_steps": 4,
            "lr": 1e-4,
            "wandb": True,
            "project": "rf-detr-flir",
            "run": "train_rf_detr_with_wandb"
        }
    )

    model = RFDETRBase()
    train_config = dict(wandb.config)

    print("WandB run initialized:", wandb.run.name)

    model.train(**train_config)

if __name__ == "__main__":
    main()
