import faiss
import polars as pl
import torch
from fastapi import FastAPI
from retrieval import gen_close_molformulas_from_seed

DATA = pl.read_parquet("./filtered_pubchem.parquet")

app = FastAPI()

RESOURCES = {
    "molecular_formulas": DATA["molecular_formula"].to_numpy(),
    "smiles": DATA["smiles"].to_numpy(),
    "index": faiss.read_index("/pubchem_vector_db_hnsw_pq.index"),
}
RESOURCES["index"].hnsw.efSearch = 128000
NUM_NEIGHBORS = 100_000


@app.post("/")
def get_smiles(request: dict) -> dict:
    mf = request["mf"]
    spectrum_embedding = request["spectrum_embedding"]
    distances, indices = RESOURCES["index"].search(torch.tensor(spectrum_embedding).unsqueeze(0), NUM_NEIGHBORS)

    def find_indices(arr, search_lst):
        return [i for i, formula in enumerate(arr.flatten()) if formula in set(search_lst)]

    close_mol_formulas = gen_close_molformulas_from_seed(mf)

    indices_correct_formula = find_indices(RESOURCES["molecular_formulas"], close_mol_formulas)
    final_smiles = RESOURCES["smiles"][indices_correct_formula].flatten().tolist()[:2048]
    if final_smiles == []:
        final_smiles = RESOURCES["smiles"][indices].flatten().tolist()[:2048]
    return {"smiles": final_smiles}
