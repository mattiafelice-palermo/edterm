import panedr
import pandas as pd

def load_data(file_path):
    try:
        df = panedr.edr_to_df(file_path)
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()  # return empty dataframe on failure
