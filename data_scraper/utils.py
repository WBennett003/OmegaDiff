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


def process_pdb(pdb, dir='datasets/pdb/', max_seq=1280, accension=''):
    xyz = np.zeros((max_seq,3))
    ss = np.zeros(max_seq) # 0 (none), 1-10 (helices), 
    chain_mask = np.zeros(max_seq)
    chains = { #example 
            '[START]' : {
            'start' : 0,
            'end' : 0,
            'shift' : 0,
        }}
    
    chain_idx = 0
    current_shift = 0

    with open(dir+pdb+'.pdb', 'r') as f:
        for line in f.readlines:
            if line[:5] == 'HELIX': #helix is 1
                i = line[20:25] - 1 #residues starts at 1 so -1 to return idx to 0
                j = line[33:37] - 1
                helix_class = line[38:49] # there are 10 helix classes
                ss[i:j] = helix_class

            elif line[:5] == 'SHEET': # sheet is 2
                i = int(line[22:26])
                j = int(line[33:36])
                sense = int(line[38:40]) #11 is antiparallel, 12 is start stand, 13 is parallel
                ss[i:j] = 12 + sense

            elif line[:3] == 'ATOM' and line[14:16] == 'CA':
                i = int(line[22:25])
                x = float(line[33:39])
                y = float(line[41:47])
                z = float(line[49:55])
                chain_ids = line[21]
                if chain_ids in chains.keys():
                    i = i + chains[chain_ids]['shift'] - chains[chain_ids]['start']
                    chain_idx = chains.keys().index(chain_ids)
                    xyz[i, 0] = x
                    xyz[i, 1] = y
                    xyz[i, 2] = z
                    chain_mask[i] = chain_idx


            elif line[:5] == 'DBREF':
                chain = line[12]
                start = int(line[13:18])
                end = int(line[18:23])
                acc = line[33:39]
                db = line[26:29]
                if db =='UNP' and acc == accension:
                    current_shift += end-start + 1
                    chains[chain] = {
                            'start': start,
                            'end' : end,
                            'shift' : current_shift
                        }
                    