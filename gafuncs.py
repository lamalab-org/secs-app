import json
from collections.abc import Callable
from random import Random
from typing import TypeVar

import numpy as np
from config import CACHE_DIR
from rdkit import Chem

from molbind.utils.spec2struct import smiles_to_molecular_formula

# Generic types for inputs and outputs
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class CachedFunction:
    """Function which caches previously computed values to avoid repeat computation."""

    def __init__(
        self,
        f: Callable,
        spectra_hash: str,
        initial_population: list[str],
    ):
        """Init function

        :type f: callable
        :param f: The function to be evaluated.
        :type spectra_hash: str
        :param spectra_hash: A hash identifying the spectra.
        :type initial_population: list[str]
        :param initial_population: A list of initial SMILES strings.
        """
        self._f = f
        self._batch_f_eval = f  # Assuming the function f handles batch evaluation
        self.spectra_hash = spectra_hash
        self.cache = {}
        self.initial_population = initial_population

    def eval_batch(self, inputs: list[str]) -> list[float]:
        """Evaluates the function on a batch of inputs, using the cache."""
        # Eval function at non-cached inputs
        inputs_not_cached = [x for x in inputs if x not in self.cache]
        if inputs_not_cached:
            outputs_not_cached = self._batch_f_eval(inputs_not_cached)
            # Add new values to cache
            for x, y in zip(inputs_not_cached, outputs_not_cached, strict=False):
                self.cache[x] = y

        # --- Start of modification ---

        # Create a list of all items in the cache
        all_items = [
            {
                "smiles": k,
                "score": v,
                "molecular_formula": smiles_to_molecular_formula(k),
                "retrieved": k in self.initial_population,
            }
            for k, v in self.cache.items()
        ]  # TODO: try to limit the size of the cache

        # Sort the list of items by score in descending order (higher score is better)
        sorted_items = sorted(all_items, key=lambda item: item["score"], reverse=True)

        # Take the top 512 items from the sorted list
        top_128_items = sorted_items[:128]

        with open(f"{CACHE_DIR!s}/{self.spectra_hash}_cache.json", "w") as f:
            json.dump(top_128_items, f)

        return [self.cache[x] for x in inputs]

    def eval_non_batch(self, inputs: str) -> float:
        """Evaluates the function on a single input, using the cache."""
        # This is a placeholder for how a non-batch evaluation might work
        if inputs not in self.cache:
            self.cache[inputs] = self._f([inputs])[0]
        return self.cache[inputs]

    def __call__(self, inputs, batch=True):
        """Allows the class instance to be called like a function."""
        # Ensure it is in batch form
        return self.eval_batch(inputs) if batch else self.eval_non_batch(inputs)


class CachedBatchFunction(CachedFunction):
    def _batch_f_eval(self, input_list):
        return self._f(input_list)


def get_number_of_topologically_distinct_atoms(smiles: str, atomic_number: int = 1):
    """Return the number of unique `element` environments based on environmental topology.
    This corresponds to the number of peaks one could maximally observe in an NMR spectrum.
    Args:
        smiles (str): SMILES string
        atomic_number (int, optional): Atomic number. Defaults to 1.

    Returns:
        int: Number of unique environments.

    Raises:
        ValueError: If not a valid SMILES string

    Example:
        >>> get_number_of_topologically_distinct_atoms("CCO", 1)
        3

        >>> get_number_of_topologically_distinct_atoms("CCO", 6)
        2
    """

    try:
        molecule = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(molecule) if atomic_number == 1 else molecule
        # Get unique canonical atom rankings
        atom_ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

        # Select the unique element environments
        atom_ranks = np.array(atom_ranks)

        # Atom indices
        atom_indices = np.array(
            [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() == atomic_number
            ]
        )
        return len(set(atom_ranks[atom_indices]))
    except (TypeError, ValueError, AttributeError, IndexError):
        return len(smiles)


def smiles_is_radical_or_is_charged_or_has_wrong_valence(smiles: str) -> bool:
    """
    Determines if a SMILES string represents a radical, charged molecule, or has wrong valence.

    Args:
        smiles (str): SMILES string representation of a molecule

    Returns:
        bool: True if the molecule is a radical OR charged OR has wrong valence, False otherwise
    """
    try:
        # Parse the SMILES string into a molecule object - without sanitization first
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        # Return False if SMILES is invalid
        if mol is None:
            return False

        # Check 1: Overall charge neutrality (before adding hydrogens)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if total_charge != 0:
            return True  # Molecule is charged

        # Check 2: Valence validity - try to sanitize
        try:
            # This will raise an exception if valence is invalid
            Chem.SanitizeMol(mol)
        except Exception:
            return True  # Molecule has wrong valence

        # Add hydrogens after sanitization succeeds
        mol = Chem.AddHs(mol)

        # Check 3: Unpaired electrons (radicals)
        for atom in mol.GetAtoms():
            # Get the number of radical electrons (unpaired electrons)
            num_radical_electrons = atom.GetNumRadicalElectrons()

            # If any atom has unpaired electrons, it's a radical
            if num_radical_electrons > 0:
                return True  # Molecule is a radical

        return False  # Molecule is neutral, has valid valence, and no radicals

    except Exception:
        # Return True for any parsing errors (likely invalid structures)
        return True
