import gdown
import os
import pandas as pd
from datetime import datetime
from natsort import natsorted
from utils.yamlparser import YamlParser

if __name__ == "__main__" :

    config_file = "/Projects/configs/config.yaml"
    update_path = "/Projects/update"
    cfg = YamlParser(config_file)
    old_df = pd.read_csv(cfg["DATA_CORPUS"]["data_csv"])

    f = natsorted(os.listdir("/Projects/update"))[-1]

    new_df_path = os.path.join(update_path, f)

    new_df = pd.read_csv(new_df_path)
    new_df.dropna(inplace=True)
    

    new_df = new_df.rename(columns={"intent" : "Intents", "question" : "Keys", "answer" :"Values"})
    new_df["Keys_vector"] = 0
    new_df = new_df.drop(columns=["time", "probability", "status"])

    df = old_df.append(new_df)
    df.to_csv("/Projects/configs/data_corpus_v2.csv",index=False ,float_format='%g')
    print("Add update done !")
