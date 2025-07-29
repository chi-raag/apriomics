import pandas as pd

from chembridge import map_metabolites

# Load the dataset
data = pd.read_csv(
    "apriomics-jsm/data/MTBLS1_files/m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv",
    sep="\t",
)

metadata = pd.read_csv("apriomics-jsm/data/MTBLS1_files/s_MTBLS1.txt", sep="\t")

data["database_identifier"].dropna().tolist()

result = map_metabolites(data["database_identifier"], target="hmdb")
data["hmdb_id"] = result.df()["hmdb_id"]

filtered_data = data[data["hmdb_id"].notna()]
