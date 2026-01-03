import pandas as pd


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Loads an uploaded CSV file into a pandas DataFrame.
    Tries UTF-8 first, falls back to latin-1 if needed.
    """
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")