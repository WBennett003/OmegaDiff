import os
import requests
import numpy as np
import json

from rdkit import Chem

from tokeniser import Element_Tokeniser, Amino_Acid_Tokeniser

ELEMENT_TOKENISER = Element_Tokeniser()
AA_TOKENISER = Amino_Acid_Tokeniser()

class smile_converter():
    def __init__(self, file_dir='datasets/smiles_cache.json'):
        self.file_dir = file_dir

        with open(file_dir, 'r+') as f:
            self.map = json.load(f)

    def __getitem__(self, chEBI):
        if chEBI in self.map.keys():
            return self.map[chEBI]
        elif chEBI in self.map['missing']:
            return ''
        else:
            smile = get_molecule(chEBI)
            if smile != '':
                self.map[chEBI] = smile
                self.update()
                return smile
            else:
                self.map['missing'].append(chEBI)
                self.update()
                return ''

    def update(self):
        with open(self.file_dir, 'w+') as f:
            json.dump(self.map, f)

SMILES = smile_converter()

def get_molecule(chebi_id):
    chem_url = f"https://www.ebi.ac.uk/chebi/saveStructure.do?defaultImage=true&chebiId={chebi_id}&imageId=0"
    try:
        data = requests.get(chem_url).text
        mol = Chem.MolFromMolBlock(data)
        if mol is not None:
            return Chem.MolToSmiles(mol)   
        else:
            return ''
    except Exception as e:
        print(f"Error, no chem {chebi_id}  || {e}")
        return ''
    

def standise_mols(atoms, bonds, max_n):
    blank_atoms = np.zeros(max_n)
    blank_bonds = np.zeros((max_n, max_n))
    
    count = 0

    for i, a in enumerate(atoms):
        n_atoms = len(a) + count
        if n_atoms > max_n:
            return [], []

        b = bonds[i]

        blank_atoms[count:n_atoms] = ELEMENT_TOKENISER.tokenise(a)
        blank_bonds[count:n_atoms, count:n_atoms] = b
        count += n_atoms

    return blank_atoms, blank_bonds

def process_mols(reactants, products):
    reactant_smiles = [SMILES[id] for id in reactants] 
    product_smiles = [SMILES[id] for id in products] 
    
    if '' in reactant_smiles or '' in product_smiles:
        return '', '', ''

    reactant_half = '.'.join(reactant_smiles)
    product_half = '.'.join(product_smiles)
    return reactant_half, product_half, reactant_half + '>>' + product_half

def AminoAcid_tokenise(sequence, max_length):
    if len(sequence) > max_length:
        return np.array([])
    x = [AA_TOKENISER.tokenise(a) for a in sequence]
    blank = np.zeros(max_length)
    blank[:len(x)] = x
    return blank