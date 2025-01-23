"""Functions for visualization of molecular data."""

def highlight_removed_atom(original_mol, sub_mol, width=500, height=500):
    """Highlight the atom removed from original_mol to create sub_mol, including its bonds."""
    from rdkit.Chem.Draw import rdMolDraw2D

    # Get the substructure match (mapping from sub_mol atoms to original_mol atoms)
    match = original_mol.GetSubstructMatch(sub_mol)
    if not match:
        # The sub_mol is not a substructure of original_mol
        raise ValueError("sub_mol is not a substructure of original_mol")

    matched_atom_indices = set(match)
    all_atom_indices = set(range(original_mol.GetNumAtoms()))

    # Identify the removed atom(s)
    removed_atoms = list(all_atom_indices - matched_atom_indices)
    if len(removed_atoms) != 1:
        # Expecting exactly one removed atom
        raise ValueError(f"Expected one removed atom, found {len(removed_atoms)}")

    removed_atom_idx = removed_atoms[0]

    # Identify bonds connected to the removed atom
    removed_bonds = []
    for bond in original_mol.GetBonds():
        if (bond.GetBeginAtomIdx() == removed_atom_idx or
            bond.GetEndAtomIdx() == removed_atom_idx):
            removed_bonds.append(bond.GetIdx())

    # Set up drawing options
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d.drawOptions()
    # Optional: Customize atom and bond highlights
    highlight_atom_colors = {removed_atom_idx: (0.0, 1.0, 1.0)}  # Red color
    highlight_bond_colors = {idx: (0.0, 1.0, 1.0) for idx in removed_bonds}

    # Draw the molecule with highlighted atom and bonds
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        original_mol,
        highlightAtoms=[removed_atom_idx],
        highlightBonds=removed_bonds,
        highlightAtomColors=highlight_atom_colors,
        highlightBondColors=highlight_bond_colors,
    )

    # Finish drawing and return SVG
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg

def extract_svg_content(svg_string):
    """
    Extracts the content inside the <svg>...</svg> tags from an SVG string.
    """
    import re
    svg_content = re.search(r'<svg[^>]*>(.*?)</svg>', svg_string, re.DOTALL).group(1)
    return svg_content

def combine_svg_strings(svg_strings, output_file, height=500, width=500):
    # Read SVG strings
    svg_contents = []
    for svg_string in svg_strings:
        svg_contents.append(extract_svg_content(svg_string))
    
    # Create combined SVG content
    combined_svg = '<svg height="' + str(height) + '" width="' + str(len(svg_strings)*width) + '">'
    x_offset = 0
    for i, svg in enumerate(svg_contents):
        # Adjust x_offset to position each SVG side by side
        combined_svg += f'<g transform="translate({x_offset}, 0)">{svg}</g>'
        x_offset += width
    
    combined_svg += '</svg>'
    
    # Save combined SVG to output file
    with open(output_file, 'w') as f:
        f.write(combined_svg)
    
    print(f"Combined SVG saved to {output_file}.")
