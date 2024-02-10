import pandas as pd
import numpy as np
import os


def read_excel_to_dataframe() -> np.ndarray:
    """
    Reading data from an Excel file.
    """
    # Root directory:
    root_dir = os.path.dirname( os.path.abspath(__file__))

    # Path and Excel sheet name:
    file_path = os.path.join(root_dir, 'data.xlsx')
    sheet_names = ['data']

    dfs = pd.read_excel(file_path, sheet_name=sheet_names, header=None)

    # Decomposing excel frames:
    decomposed = []
    for df in dfs.values():

        # Replace nan values with string (easier to check for):
        df = df.replace(np.nan, "NAN")

        if np.ndim(df.values) > 1:
            decomposed.append(df.values[1:, 1:])
        else:
            decomposed.append(df.values[1:])
    return np.array(decomposed[0])

