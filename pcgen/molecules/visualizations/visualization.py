"""Functions for visualization of molecular data."""

def generate_non_substructure_vis(mol, sub, height=500, width=500):
    """Higlight non-substructure atoms and bonds in a molecule as a svg."""
    from rdkit.Chem.Draw import rdMolDraw2D
    
    # Get substructure match
    hit_ats = set(mol.GetSubstructMatch(sub))
    all_ats = set(range(mol.GetNumAtoms()))
    non_hit_ats = list(all_ats - hit_ats)
    
    # Get bonds not in substructure match
    non_hit_bonds = []
    for bond in mol.GetBonds():
        aid1 = bond.GetBeginAtomIdx()
        aid2 = bond.GetEndAtomIdx()
        if aid1 in non_hit_ats or aid2 in non_hit_ats:
            non_hit_bonds.append(bond.GetIdx())
    
    # Create MolDraw2DSVG object
    d = rdMolDraw2D.MolDraw2DSVG(height, width)
    
    # Draw molecule with highlighted non-substructure
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=non_hit_ats, highlightBonds=non_hit_bonds)
    
    # Finish drawing
    d.FinishDrawing()
    
    # Get SVG string
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
