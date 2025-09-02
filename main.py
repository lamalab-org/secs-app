import random
from functools import partial

import numpy as np
import requests
import torch
from gafuncs import CachedBatchFunction, smiles_is_radical_or_is_charged_or_has_wrong_valence
from loguru import logger
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.preconfigured_gas import default_ga
from prune import gpu_encode_smiles_variable, load_models_dict
from rdkit import Chem
from retrieval import SimpleMoleculeAnalyzer
from torch.nn.functional import cosine_similarity as torch_cosine_similarity

from molbind.utils.spec2struct import get_atom_counts_from_formula, smiles_to_molecular_formula


def compute_individual_atom_counts(individual: str) -> dict | None:
    mol = Chem.MolFromSmiles(individual)
    if not mol:
        logger.warning(f"Invalid SMILES for atom count: {individual}")
        return None
    mol = Chem.AddHs(mol)
    counts = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return counts


def calculate_mf_penalty(smi: str, atom_counts_orig: dict) -> float:
    """Calculates the molecular formula penalty for a single SMILES string."""
    counts_i = compute_individual_atom_counts(smi)
    if not counts_i:
        return -1000.0  # Heavy penalty for invalid SMILES

    total_orig_atoms = sum(atom_counts_orig.values()) if atom_counts_orig else 1.0
    if total_orig_atoms == 0:
        total_orig_atoms = 1.0

    penalty = sum(abs(counts_i.get(s, 0) - atom_counts_orig.get(s, 0)) for s in set(counts_i) | set(atom_counts_orig))
    return -penalty / total_orig_atoms


def reward_function_ga(
    individuals: list[str],
    ga_models: dict,
    target_1D_embs: dict,
    atom_counts_orig: dict,
) -> np.array:
    if not individuals:
        return np.array([])

    cand_smiles_embs = gpu_encode_smiles_variable(individuals, ga_models)

    mf_loss = np.array([calculate_mf_penalty(smi, atom_counts_orig) for smi in individuals])

    scores_all_mods_np, num_ok_mods = [], 0
    for spec, target_emb_1D_gpu in target_1D_embs.items():  # target_emb is 1D (D,)
        if spec not in cand_smiles_embs or cand_smiles_embs[spec] is None or cand_smiles_embs[spec].nelement() == 0:
            continue  # No candidate embeddings for this modality

        cand_embs_mod_gpu = cand_smiles_embs[spec]  # (N, D_mod)

        sims_gpu = torch_cosine_similarity(
            target_emb_1D_gpu.unsqueeze(0).to("cpu"), cand_embs_mod_gpu.to("cpu"), dim=1
        )  # (1,D) vs (N,D) -> (N,)
        scores_all_mods_np.append(sims_gpu.cpu().numpy())
        num_ok_mods += 1

    if num_ok_mods == 0:
        return mf_loss  # Only MF penalty if no spectral scores

    avg_cosine_sim = np.mean(np.array(scores_all_mods_np), axis=0)
    is_radical_or_charged = np.array([smiles_is_radical_or_is_charged_or_has_wrong_valence(smi) for smi in individuals])
    return avg_cosine_sim + mf_loss - is_radical_or_charged


def calculate_detailed_scores(smi: str, models: dict, target_1D_embs: dict, atom_counts_orig: dict) -> dict:
    """Calculates and returns a dictionary of detailed scores for a single SMILES."""
    if smi == "N/A" or not smi:
        return {"mf_penalty": -1000.0}

    scores = {}
    # 1. Calculate Molecular Formula Penalty
    scores["mf_penalty"] = calculate_mf_penalty(smi, atom_counts_orig)

    # 2. Calculate Modality-specific Cosine Similarities
    smi_emb_dict = gpu_encode_smiles_variable([smi], models)
    for spec, target_emb in target_1D_embs.items():
        score_key = f"{spec}_cosine_sim"
        if spec not in smi_emb_dict or smi_emb_dict[spec] is None or smi_emb_dict[spec].nelement() == 0:
            scores[score_key] = 0.0
            continue

        cand_emb = smi_emb_dict[spec]
        sim = torch_cosine_similarity(target_emb.unsqueeze(0), cand_emb, dim=1)
        scores[score_key] = sim.item()

    return scores


def run_ga_instance(
    initial_pop: list[str],
    models: dict,
    atom_counts_orig: dict,
    target_1D_embs: dict,
    ga_params: dict,
    idx: int = 0,
):
    # Unpack GA parameters
    # init_pop = ga_params["initial_population_size_from_pruning"]
    gens = ga_params["generations"]
    offspring = ga_params["offspring_size"]
    pop_ga = ga_params["population_size"]
    seed_val = ga_params["seed"]
    frac_mutate = ga_params["frac_graph_ga_mutate"]

    reward_f = partial(reward_function_ga, ga_models=models, target_1D_embs=target_1D_embs, atom_counts_orig=atom_counts_orig)

    ga_logger = logger

    ga_res = default_ga(
        starting_population_smiles=initial_pop,
        scoring_function=CachedBatchFunction(reward_f),
        max_generations=gens,
        offspring_size=offspring,
        population_size=pop_ga,
        logger=ga_logger,
        rng=random.Random(seed_val),
        offspring_gen_func=partial(graph_ga_blended_generation, frac_graph_ga_mutate=frac_mutate),
    )

    best_sc, best_smi = max(ga_res.population, key=lambda x: x[0]) if ga_res.population else (-float("inf"), "N/A")
    logger.info(f"GA Best for idx {idx}: {best_smi} (score {best_sc:.4f})")

    return ga_res.population


def spec2struct(
    mf: str,
    spectrum: dict[str, list[float]],
    model: str = "residual",
    configs_path: str = "configs",
    ga_ir_exp: str | None = None,
    # ga_cnmr_exp: str | None = "configs/cnmr",
    ga_cnmr_exp: str | None = None,
    ga_hnmr_exp: str | None = "test/hnmr_augment_finetune_residual",
    ga_hsqc_exp: str | None = None,
    seed: int = 42,
    init_pop_ga: int = 512,
    frac_graph_ga_mutate: float = 0.3,
    gens_ga: int = 10,
    offspring_ga: int = 1024,
    pop_ga: int = 512,
    initial_environment: dict[str, str] | None = None,
):
    dict_models = {
        "residual": "test/hnmr_augment_finetune_residual",
        "regular": "test/hnmr_augment_finetune",
    }
    ga_hnmr_exp = dict_models.get(model, ga_hnmr_exp)
    atom_count_dict = get_atom_counts_from_formula(mf)

    def canonicalize_molecular_formula(atom_count_dict):
        # CxHy then alphabetically sorted elements
        carbon_count = atom_count_dict.get("C", 0)
        hydrogen_count = atom_count_dict.get("H", 0)
        carbon_hydrogen_mol_formula = f"C{carbon_count}H{hydrogen_count}" if carbon_count > 0 or hydrogen_count > 0 else ""

        return carbon_hydrogen_mol_formula + "".join(
            sorted(
                f"{el}{count}" if count > 1 else el for el, count in atom_count_dict.items() if count > 0 and el not in ("C", "H")
            )
        )

    mf = canonicalize_molecular_formula(atom_count_dict)

    ga_model_exps = {"ir": ga_ir_exp, "cnmr": ga_cnmr_exp, "hnmr": ga_hnmr_exp, "hsqc": ga_hsqc_exp}

    ga_params = {
        "seed": seed,
        "initial_population_size_from_pruning": init_pop_ga,
        "generations": gens_ga,
        "offspring_size": offspring_ga,
        "population_size": pop_ga,
        "frac_graph_ga_mutate": frac_graph_ga_mutate,
        "model_experiments": {k: v for k, v in ga_model_exps.items() if v},
        # "raw_embedding_paths": {k: v for k, v in ga_raw_emb_paths_map.items() if v},
        # "dataset_path": dataset_path,
    }

    ga_models_for_scoring = load_models_dict(configs_path, ga_model_exps)
    active_ga_model_modalities = [m for m, model in ga_models_for_scoring.items() if model]
    final_ga_models_to_use = {m: ga_models_for_scoring[m] for m in active_ga_model_modalities if m in ga_models_for_scoring}
    logger.info(f"GA models loaded: {final_ga_models_to_use.keys()}")
    # if not active_ga_model_modalities:
    #     logger.error("No GA models loaded. Exiting.")
    #     return
    # logger.info(f"GA scoring models: {active_ga_model_modalities}")

    analyzer_user_active_spectra = [m for m in SimpleMoleculeAnalyzer.ALL_SPECTRA_TYPES if locals().get(f"analyzer_{m}_exp")]
    analyzer = SimpleMoleculeAnalyzer(
        models_config_path=configs_path,
        active_spectra=analyzer_user_active_spectra,
    )
    analyzer.load_models(
        ir_experiment=ga_ir_exp, cnmr_experiment=ga_cnmr_exp, hnmr_experiment=ga_hnmr_exp, hsqc_experiment=ga_hsqc_exp
    )

    if spectrum["x"][0] < 9:
        spectrum["y"] = spectrum["y"][::-1]

    spectrum_as_tensor = torch.tensor(spectrum["y"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spectrum_as_tensor.requires_grad = False
    spectrum_as_tensor = spectrum_as_tensor / spectrum_as_tensor.max()
    output_hnmr_embedding = analyzer.models["hnmr"].encode_modality(spectrum_as_tensor.detach(), "h_nmr")

    target_embeddings = {"hnmr": output_hnmr_embedding.squeeze(0).detach()}

    torch.nn.functional.normalize(target_embeddings["hnmr"], p=2, dim=0, out=target_embeddings["hnmr"])

    if initial_environment:
        forward_synthesis_address = "http://forward_synthesis:7998/"
        response = requests.post(
            forward_synthesis_address,
            json={"reagents": initial_environment["reagents"], "reactants": initial_environment["reactants"]},
        )
        initial_pop = response.json()["smiles"]
    else:
        spectrum_embedding = target_embeddings["hnmr"].flatten().numpy().tolist()

        docker_vector_db = "http://vectordb:7999/"

        response = requests.post(docker_vector_db, json={"mf": mf, "spectrum_embedding": spectrum_embedding})
        initial_pop = response.json()["smiles"]

    results = run_ga_instance(
        initial_pop=initial_pop,
        models=final_ga_models_to_use,
        atom_counts_orig=get_atom_counts_from_formula(mf),
        target_1D_embs=target_embeddings,
        ga_params=ga_params,
    )

    results_dict = {v: k for k, v in results}

    return [
        {
            "smiles": k,
            "score": v,
            "molecular_formula": smiles_to_molecular_formula(k),
            "retrieved": k in initial_pop,
        }
        for k, v in results_dict.items()
    ]
