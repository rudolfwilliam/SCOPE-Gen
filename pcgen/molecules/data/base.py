import torch
from collections import Counter

def squeeze(tuple):
    """Squeeze empty dimensions from a (X, E) tuple (similar to squeeze() in PyTorch)."""
    return tuple[0].squeeze(0), tuple[1].squeeze(0)

def subtract_arrays(large_array, small_array):
    # Create a dictionary to count occurrences of each element in the smaller array
    small_count = {}
    for element in small_array:
        if element.item() in small_count:
            small_count[element.item()] += 1
        else:
            small_count[element.item()] = 1
    
    # Resultant array which will store the difference
    result = []
    
    # Iterate through the larger array and subtract the elements based on the counts
    for element in large_array:
        if element.item() in small_count and small_count[element.item()] > 0:
            small_count[element.item()] -= 1
        else:
            result.append(element)
    
    # Convert the result list back to a PyTorch tensor
    result_tensor = torch.tensor(result)
    
    return result_tensor

def count_atom_types(molecule):
    # Initialize a counter to keep track of atom types
    atom_counter = Counter()
    
    # Iterate over the atoms in the molecule and count the occurrences of each atom type
    for atom in molecule.GetAtoms():
        atom_type = atom.GetSymbol()
        atom_counter[atom_type] += 1
    
    return atom_counter
