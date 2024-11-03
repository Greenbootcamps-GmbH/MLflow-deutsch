import mlflow

# Definiert den Namen des Experiments in MLflow und den Entry-Point des MLproject

# experiment_name = "ElasticNet"  
# entry_point = "Training-Elastic-Net"   
experiment_name = "Ridge"
entry_point = "Training-Ridge"

# Startet ein MLflow-Projekt basierend auf dem angegebenen Einstiegspunkt und Experimentnamen
mlflow.projects.run(
    uri=".",                     # URI des Projekts (lokales Verzeichnis, angegeben als ".")
    entry_point=entry_point,     # Einstiegspunkt, der im MLproject als 'Training' definiert ist
    experiment_name=experiment_name,  # Name des Experiments, um Ergebnisse in MLflow zu loggen
    env_manager="local"          # Verwendet die lokale Umgebung statt eine neue zu erstellen
)
