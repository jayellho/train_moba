import optuna
import pandas as pd

# 1. Load the study from the SQLite file
study = optuna.load_study(
    study_name="whisper_hptune",
    storage="sqlite:///whisper_hptune.db"
)

# 2. Print the best parameters and best objective value
print("Best params:", study.best_params)
print("Best WER:   ", study.best_value)

# 3. (Optional) Get a full DataFrame of all trials
df = study.trials_dataframe()
df.to_excel("optuna_trials.xlsx")
print(df)