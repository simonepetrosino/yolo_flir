import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend per salvare i grafici senza GUI
import matplotlib.pyplot as plt
import os

# Lista file CSV da caricare
files = {
    "yolov8s": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov8s.csv",
    "yolov9s": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov9s.csv",
    "yolov10s": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov10s.csv",
    "yolov11s": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov11s.csv",
    "yolov12s": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov12s.csv",
    "yolov11s_aug9": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov11s_aug9.csv",
    "yolov11s_aug10": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov11s_aug10.csv",
    "yolov11s_aug11": r"C:\Users\simo5\Desktop\yoloprovalocale\best validations\yolov11s_aug11.csv",
}

# Metriche globali da confrontare
metrics_global = ['precision', 'recall', 'mAP50', 'mAP50-95']

# Classi per cui generare i grafici mAP50 per singola classe
selected_classes = ['person', 'bike', 'motor', 'car', 'bus', 'truck', 'light', 'hydrant', 'sign', 'skateboard', 'stroller', 'other vehicle' ]

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Normalizza nomi colonne (minuscolo e sostituzioni)
    df.columns = [col.lower().strip() for col in df.columns]
    # Rinomi espliciti (se ci sono vecchi nomi da correggere)
    rename_map = {
        'box(p)': 'precision',
        'r': 'recall',
        'map50-95': 'mAP50-95',
        'map50': 'mAP50',
        'class': 'class',
        'images': 'images',
        'instances': 'instances',
    }
    df.rename(columns=rename_map, inplace=True)

    # Normalizza nomi classi (minuscolo)
    if 'class' in df.columns:
        df['class'] = df['class'].str.lower()
    else:
        raise ValueError(f"Colonna 'class' non trovata nel file {file_path}")

    return df

# Carica dati
data = {}
for model, path in files.items():
    if os.path.isfile(path):
        try:
            data[model] = load_data(path)
        except Exception as e:
            print(f"❌ Errore caricando {model}: {e}")
    else:
        print(f"❌ File non trovato: {path}")

if not data:
    raise ValueError("Nessun file CSV valido trovato.")

# Grafici globali (media metrica per modello, classe 'all')
for metric in metrics_global:
    plt.figure(figsize=(10, 6))
    values = []
    models = []
    for model, df in data.items():
        if metric in df.columns:
            all_row = df[df['class'] == 'all']
            if not all_row.empty:
                val = all_row[metric].values[0]
                values.append(val)
                models.append(model)
            else:
                print(f"⚠️ {model}: classe 'all' non trovata per la metrica {metric}")
        else:
            print(f"⚠️ Metrica {metric} non trovata nel dataset {model}")

    if values:
        plt.bar(models, values, color='skyblue')
        plt.title(f'Confronto globale {metric}')
        plt.ylabel(metric)
        plt.xlabel('Modello')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'global_{metric}.png')
        plt.close()
    else:
        print(f"⚠️ Nessun dato valido per la metrica globale {metric}")

# Grafici per singola classe (solo mAP50, per selected_classes)
for cls in selected_classes:
    plt.figure(figsize=(10, 6))
    values = []
    models = []
    for model, df in data.items():
        if 'mAP50' in df.columns:
            filtered = df[df['class'] == cls]
            if not filtered.empty:
                val = filtered['mAP50'].values[0]  # Uso esatto colonna con M maiuscola
                values.append(val)
                models.append(model)
            else:
                print(f"⚠️ {model}: mAP50 non trovata per classe '{cls}'")
        else:
            print(f"⚠️ {model}: metrica mAP50 non trovata")

    if values:
        plt.bar(models, values, color='orange')
        plt.title(f'Confronto mAP50 per classe "{cls}"')
        plt.ylabel('mAP50')
        plt.xlabel('Modello')
        plt.ylim(0, 1)  # mAP50 sempre tra 0 e 1
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'class_{cls}_mAP50.png')
        plt.close()
    else:
        print(f"⚠️ Nessun dato valido per classe {cls} su mAP50")


    if values:
        plt.bar(models, values, color='orange')
        plt.title(f'Confronto mAP50 per classe "{cls}"')
        plt.ylabel('mAP50')
        plt.xlabel('Modello')
        plt.ylim(0, 1)  # mAP50 sempre tra 0 e 1
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'class_{cls}_mAP50.png')
        plt.close()
    else:
        print(f"⚠️ Nessun dato valido per classe {cls} su mAP50")

print("Grafici generati e salvati.")
