import wandb
import time
from ultralytics import YOLO

# Variabile globale per salvare l'epoca corrente e il tempo per epoca
CURRENT_EPOCH = 0
EPOCH_START_TIME = 0

class WandbLoggerCallback:
    """Callback per loggare metriche su W&B."""

    def __init__(self):
        wandb.login()

    def on_train_epoch_end(self, trainer):
        global CURRENT_EPOCH
        global EPOCH_START_TIME
        CURRENT_EPOCH = trainer.epoch

        # Calcola e logga il tempo per epoca
        epoch_time = time.time() - EPOCH_START_TIME
        EPOCH_START_TIME = time.time() 

        # Loss del training
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            box_loss, cls_loss, dfl_loss = trainer.loss_items[:3]
        else:
            box_loss = cls_loss = dfl_loss = 0

        wandb.log({
            "epoch": CURRENT_EPOCH,
            "train/box_loss": box_loss,
            "train/cls_loss": cls_loss,
            "train/dfl_loss": dfl_loss,
            "train/lr": trainer.optimizer.param_groups[0]["lr"],  # aggiunto learning rate
            "train/epoch_time": epoch_time,  # Tempo per epoca
        })

    def on_val_end(self, validator):
        global CURRENT_EPOCH

        # Metriche di validazione corrette
        if hasattr(validator.metrics, 'box') and validator.metrics.box is not None:
            precision = validator.metrics.box.p.mean().item()
            recall = validator.metrics.box.r.mean().item()
            map50 = validator.metrics.box.map50.item()
            map50_95 = validator.metrics.box.map.item()

            # Calcolo F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = map50 = map50_95 = f1_score = 0

        wandb.log({
            "epoch": CURRENT_EPOCH,
            "val/precision": precision,
            "val/recall": recall,
            "val/F1_score": f1_score,  # Aggiunto F1 Score
            "val/mAP_50": map50,
            "val/mAP_50_95": map50_95,
        })

def main():
    MODEL = 'yolov8n.pt'
    DATA = ''
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH = 16
    PATIENCE = 10
    PROJECT_NAME = ''
    RUN_NAME = ''

    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    model = YOLO(MODEL)

    logger_callback = WandbLoggerCallback()
    model.add_callback('on_train_epoch_end', logger_callback.on_train_epoch_end)
    model.add_callback('on_val_end', logger_callback.on_val_end)

    # Salva il tempo di inizio epoca
    global EPOCH_START_TIME
    EPOCH_START_TIME = time.time()

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        verbose=True,
        val=True,
        save=True,
        device=0,
        plots=True,
    )

    wandb.finish()

if __name__ == '__main__':
    main()
