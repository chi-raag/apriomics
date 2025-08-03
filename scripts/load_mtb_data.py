# %%
import pandas as pd

# %%
mtb_data = pd.read_csv("../docs/examples/data/mtb.tsv", sep="\t")

# %%
sample_names = mtb_data["Sample"]
groups = ["con" if "CON" in x else "kd" for x in sample_names]
# %%
metabolite_names = mtb_data.columns.tolist()[1:]
intensity_data = mtb_data.iloc[:, 1:]
# %%
