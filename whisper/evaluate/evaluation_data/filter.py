import pandas as pd
import os

df = pd.read_excel("NSC Part 4 Speaker Metadata.xlsx", dtype={'Speaker ID': str})
print(df)
base_path = "./Codeswitching"

valid_speakers = []
for idx, row in df.iterrows():
    speaker = row["Speaker ID"].zfill(4)
    session = str(row["Session ID"]).zfill(4)

    found = False
    for prefix in ["phns_cs", "phnd_cs"]:
        for lang in ["chn", "mly", "tml"]:
            for mic in ["Diff Room Audio", "Same Room Audio"]:
                fname = f"sur_{session}_{speaker}_{prefix}-{lang}.wav"
                fpath = os.path.join(base_path, mic, fname)
                if os.path.exists(fpath):
                    found = True
                    break
        if found:
            break

    if found:
        valid_speakers.append(idx)

df_filtered = df.loc[valid_speakers]
print(df_filtered)
df_filtered.to_excel("metadata_filtered.xlsx", index=False)
