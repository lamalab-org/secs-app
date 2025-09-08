from fastapi import FastAPI
from rdkit import Chem
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()
TOKENIZER = AutoTokenizer.from_pretrained(
    "sagawa/ReactionT5v2-forward-USPTO_MIT", return_tensors="pt"
)
MODEL = AutoModelForSeq2SeqLM.from_pretrained("sagawa/ReactionT5v2-forward-USPTO_MIT")


def is_valid_smiles(smiles):
    """
    Fast SMILES validation using RDKit.

    Args:
        smiles (str): SMILES string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return mol is not None
    except:
        return False


def get_elements_from_smiles(smiles):
    """Extract unique elements from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def has_charge(smiles):
    """Check if a molecule has an overall charge (non-zero formal charge)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # Invalid SMILES, consider as charged to filter out
    return Chem.rdmolops.GetFormalCharge(mol) != 0


@app.post("/")
def get_smiles(request: dict) -> dict:
    reagents, reactants = request["reagents"], request["reactants"]

    # Get allowed elements from reactants and reagents
    allowed_elements = set()
    allowed_elements.update(get_elements_from_smiles(reactants))
    allowed_elements.update(get_elements_from_smiles(reagents))

    inp = TOKENIZER(
        f"REACTANT: {Chem.CanonSmiles(reactants)} REAGENT: {Chem.CanonSmiles(reagents)}",
        return_tensors="pt",
    )
    output = MODEL.generate(
        **inp,
        num_beams=100,
        num_return_sequences=50,
        return_dict_in_generate=True,
        output_scores=True,
    )

    smiles_predictions = []
    for i in range(50):
        output_i = (
            TOKENIZER.decode(output["sequences"][i], skip_special_tokens=True)
            .replace(" ", "")
            .rstrip(".")
        )
        smiles_predictions.append(output_i)

    # Process and filter SMILES
    final_smiles = []
    for smi in smiles_predictions:
        if not is_valid_smiles(smi):
            continue

        # Split by dots to separate multiple compounds
        compounds = smi.split(".")

        for compound in compounds:
            compound = compound.strip()
            if not compound:  # Skip empty strings
                continue

            # Check if compound is valid
            if not is_valid_smiles(compound):
                continue

            # Check if compound has charge
            if has_charge(compound):
                continue

            # Check if compound contains only allowed elements
            compound_elements = get_elements_from_smiles(compound)
            if not compound_elements.issubset(allowed_elements):
                continue

            final_smiles.append(compound)

    # Remove duplicates while preserving order
    seen = set()
    unique_smiles = []
    for smi in final_smiles:
        canonical_smi = Chem.CanonSmiles(smi)
        if canonical_smi and canonical_smi not in seen:
            seen.add(canonical_smi)
            unique_smiles.append(canonical_smi)
    print(unique_smiles)
    return {"smiles": unique_smiles}
