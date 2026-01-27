import pandas as pd  # Pandas for reading CSVs into DataFrames


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Loads an uploaded CSV file into a pandas DataFrame.
    Tries UTF-8 first, falls back to latin-1 if needed.
    """

    # Most CSVs work with default UTF-8 reading
    try:
        return pd.read_csv(uploaded_file)

    # If encoding fails (common in older CSVs), retry with latin-1
    except UnicodeDecodeError:
        uploaded_file.seek(0)  # reset file pointer back to start before re-reading
        return pd.read_csv(uploaded_file, encoding="latin-1")
