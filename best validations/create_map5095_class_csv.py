import os
import pandas as pd

# Percorso assoluto alla cartella dove si trovano i file CSV (modifica questa riga se necessario)
folder_path = os.path.abspath(os.path.dirname(__file__))  # cartella dove si trova lo script

# Nomi dei file e dei modelli corrispondenti
model_files = {
    "yolov8s": "yolov8s.csv",
    "yolov9s": "yolov9s.csv",
    "yolov10s": "yolov10s.csv",
    "yolov11s": "yolov11s.csv",
    "yolov12s": "yolov12s.csv",
    "yolov11s_aug9": "yolov11s_aug9.csv",
    "yolov11s_aug10": "yolov11s_aug10.csv",
    "yolov11s_aug11": "yolov11s_aug11.csv",
}

# Lista per contenere i DataFrame da unire
dfs = []

for model, filename in model_files.items():
    file_path = os.path.join(folder_path, filename)
    print(f"🔍 Cerco file: {file_path}")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if "class" in df.columns and "map50-95" in df.columns:
                df = df[["class", "map50-95"]].copy()
                df.rename(columns={"map50-95": model}, inplace=True)
                dfs.append(df)
                print(f"✅ File letto correttamente: {filename}")
            else:
                print(f"⚠️ Colonne mancanti in {filename}.")
        except Exception as e:
            print(f"❌ Errore durante la lettura di {filename}: {e}")
    else:
        print(f"❌ File non trovato: {file_path}")

# Unione dei DataFrame se almeno uno è stato caricato
if dfs:
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on="class", how="outer")

    final_df.sort_values("class", inplace=True)
    final_df.set_index("class", inplace=True)

    output_path = os.path.join(folder_path, "mAP50-95_per_classe.csv")
    final_df.to_csv(output_path)
    print(f"✅ File CSV creato: {output_path}")
else:
    print("⚠️ Nessun file valido trovato. Verifica i percorsi e i nomi dei file.")
