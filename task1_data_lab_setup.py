"""Task 1: verify the Python Data Science environment."""
import sys
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

print("Data Science environment ready!")
print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)

# Small sanity check
sample = pd.DataFrame({"value": [10, 20, 30, 40]})
print("\nSample data summary:")
print(sample.describe())
