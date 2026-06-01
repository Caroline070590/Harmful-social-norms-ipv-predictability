# Auto-extracted from notebooks/12_download_hofstede_data.ipynb

# Review/reproducibility helper script. Original notebook is retained in notebooks/.



# %% Cell 0
import kagglehub

# Download latest version
path = kagglehub.dataset_download("seydakaba/hofstede-cultural-dimensions-by-country")

print("Path to dataset files:", path)


# %% Cell 1

