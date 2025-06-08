import os
import pandas as pd

# Percorso della cartella dove si trovano i file CSV
folder_path = os.path.abspath(os.path.dirname(__file__))

# Nomi dei file e dei modelli corrispondenti
model_files = {
    
}

# Lista dei dizionari con i dati per ogni modello
summary_data = []

for model, filename in model_files.items():
    file_path = os.path.join(folder_path, filename)
    print(f"🔍 Cerco file: {file_path}")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            all_row = df[df["class"] == "all"]
            if not all_row.empty:
                summary_data.append({
                    "model": model,
                    "precision": all_row["precision"].values[0],
                    "recall": all_row["recall"].values[0],
                    "map50": all_row["map50"].values[0],
                    "map50-95": all_row["map50-95"].values[0]
                })
                print(f"✅ Dati estratti per: {model}")
            else:
                print(f"⚠️ Riga con classe 'all' non trovata in {filename}.")
        except Exception as e:
            print(f"❌ Errore durante la lettura di {filename}: {e}")
    else:
        print(f"❌ File non trovato: {file_path}")

# Creazione del DataFrame finale
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index("model", inplace=True)
    output_path = os.path.join(folder_path, "yolo_all_class_summary.csv")
    summary_df.to_csv(output_path)
    print(f"✅ File CSV creato: {output_path}")
else:
    print("⚠️ Nessun dato disponibile per creare il riepilogo.")
