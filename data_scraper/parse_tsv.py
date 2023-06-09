import h5py
from tokeniser import Amino_Acid_Tokeniser
import numpy as np
import time
from drfp import DrfpEncoder
import json 


with open('datasets/cofactors.json', 'r') as f:
    cofactors_dict = json.load(f)

def padding(seq, maxlength, padding_idx=21):
    blank = np.zeros(maxlength)
    blank[len(seq):] = padding_idx
    blank[:len(seq)] = seq
    return blank

class parse_tsv:
    def __init__(self, inp_file_dir, out_file_dir, max_seq_aa=1280, reaction_embed_size=10240, max=None, hydrogens=True, root_atom=True, n_cofactors=4) -> None:
        self.inp_file_dir = inp_file_dir
        self.out_file_dir = out_file_dir
        self.max_aa_seq = max_seq_aa
        self.reaction_size = reaction_embed_size
        self.reaction_encoder = DrfpEncoder
        self.hydrogens = hydrogens
        self.root_atom = root_atom
        self.n_cofactors = n_cofactors

        self.tokeniser = Amino_Acid_Tokeniser()

        self.initalise_ds()

        self.run(max=max)

    def step_row(self, row):
        #clean row
        row = row.split('\t')
        if len(self.columns) == len(row): # if there is an an error with the row
            acc, ID, seq, length, taxid, name, ecid, eqx, cheqx, smilerxn, reactsmile, prodsmile, rhea, features, pdb_code, cofactors = row
            if int(length) < self.max_aa_seq: # if the sequence is longer than the standard length
                tokens = self.tokeniser(seq)
                tokens = padding(tokens, self.max_aa_seq)
                if min(tokens) >= 0.0:
                    # reaction = np.zeros(self.reaction_size)
                    reaction = self.reaction_encoder.encode(smilerxn, n_folded_length=self.reaction_size, radius=3, include_hydrogens=self.hydrogens, root_central_atom=self.root_atom)
                    # xyz, ss, chains = process_pdb(pdb_code, max_seq=self.max_aa_seq, accension=acc)
                    if sum(reaction) < 2:
                        xyz, ss, chains = None, None, None
                        active, tm = process_features(features, self.max_aa_seq)
                        cofactors = process_cofactors(cofactors, self.n_cofactors)
                        self.update_ds(acc, ID, seq, length, taxid, name, ecid, eqx, cheqx, smilerxn, reactsmile, prodsmile, rhea, tokens, reaction, xyz, ss, chains, active, cofactors)

    def run(self, step=5000, max=None):
        if max is None:
            with open(self.inp_file_dir, 'r') as f:
                n = len(f.readlines())
        else:
            n = max

        with open(self.inp_file_dir, 'r') as f:
            print(f"PROCESSING {n} SAMPLES!!!")
            line = f.readline()
            self.columns = line.split('\t')
            last_time = time.time()
            for i in range(n):
                if i % step == 0:
                    new_time = time.time()
                    perc = (n - i) / n
                    diff = new_time - last_time
                    ETA = new_time + perc * diff
                    print(f" {i} / {n} | {round(100-perc, 3)} | ETA {time.ctime(ETA)}")
                    last_time = new_time

                line = f.readline()
                self.step_row(line)


    def initalise_ds(self):
        string_dt = h5py.special_dtype(vlen=str)

        self.file = h5py.File(self.out_file_dir+'.h5', 'w')
        self.file.create_dataset('accension', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('id', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('sequence', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('length', (0, 1), dtype='i', chunks=True, maxshape=(None, 1))
        self.file.create_dataset('tax_id', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('name', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('ec_id', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('chebi_equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('smile_equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('reactants', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('products', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))        
        
        self.file.create_dataset('reactants_smile', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('products_smile', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('rhea_id', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('aa_seq', (0, self.max_aa_seq), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq))

        self.file.create_dataset('reaction', (0, self.reaction_size), dtype='i', chunks=True, maxshape=(None, self.reaction_size))
        # self.file.create_dataset('secondary_structure', (0, self.max_aa_seq), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq))
        # self.file.create_dataset('chain_mask', (0, self.max_aa_seq), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq))
        self.file.create_dataset('active_res', (0, self.max_aa_seq), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq))
        self.file.create_dataset('cofactors', (0, self.n_cofactors), dtype='i', chunks=True, maxshape=(None, self.n_cofactors))
        # self.file.create_dataset('coords', (0, self.max_aa_seq, 3), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq, 3))


    def update_ds(self, accension, uniprot_id, sequence, seq_len, tax_id, name, ec_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles, Rhea_id, tokenised_sequence, reaction, xyz, ss, chains, active, cofactors):
        self.save_dataset('accension', accension)
        self.save_dataset('id', uniprot_id)
        self.save_dataset('sequence', sequence)
        self.save_dataset('length', int(seq_len))
        self.save_dataset('tax_id', str(tax_id))
        self.save_dataset('name', name)
        self.save_dataset('ec_id', ec_id)
        self.save_dataset('equation', equation)
        self.save_dataset('chebi_equation', chebi_equation)
        self.save_dataset('smile_equation', smile_reaction)
        self.save_dataset('reactants_smile', reactants_smiles)
        self.save_dataset('products_smile', products_smiles)
        self.save_dataset('rhea_id', Rhea_id)
        self.save_dataset('aa_seq', tokenised_sequence)
        self.save_dataset('reaction', reaction)
        # self.save_dataset('coords', xyz)
        # self.save_dataset('secondary_structure', ss)
        # self.save_dataset('chain_mask', chains)
        self.save_dataset('active_res', active)
        self.save_dataset('cofactors', cofactors)

    def save_dataset(self, key, arr):
        self.file[key].resize((self.file[key].shape[0] + 1), axis=0)
        self.file[key][-1:] = arr


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
    next_shift = 0
    last_end = 0

    cut_chain = False

    with open(dir+pdb+'.pdb', 'r') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            if line[:5] == 'HELIX': #helix is 1
                chain_id = line[19]
                if chain_id in chains.keys():
                    shift = chains[chain_id]['shift']
                    i = int(line[20:25]) + shift# add shift to correct residue indexing as there is a padding spacer between chains
                    j = int(line[33:37]) + shift
                    helix_class = int(line[38:40]) # there are 10 helix classes
                    ss[i:j] = helix_class
                    pass

            elif line[:5] == 'SHEET': # sheet is 2
                chain_id = line[21]
                if chain_id in chains.keys():
                    shift = chains[chain_id]['shift']
                    i = int(line[22:26]) + shift
                    j = int(line[33:36]) + shift
                    sense = int(line[38:40]) #11 is antiparallel, 12 is start stand, 13 is parallel
                    ss[i:j] = 12 + sense
                    pass

            elif line[:4] == 'ATOM' and line[13:15] == 'CA':
                i = int(line[22:26]) - 1
                x = float(line[32:39])
                y = float(line[40:47])
                z = float(line[48:55])
                chain_ids = line[21]
                if chain_ids in chains.keys():
                    i = i + chains[chain_ids]['shift'] 
                    xyz[i, 0] = x
                    xyz[i, 1] = y
                    xyz[i, 2] = z
                    

            elif line[:5] == 'DBREF':
                chain = line[12]
                start = int(line[13:18]) - 1
                end = int(line[18:24])
                acc = line[33:39]
                db = line[26:29]
                if db =='UNP' and acc == accension:
                    chains[chain] = {
                            'start': start,
                            'end' : end,
                            'shift' : next_shift
                        }
                    
                    next_shift += last_end-start
                    chain_mask[start+next_shift:end+next_shift] = len(chains) - 1
                    last_end = end

    chain_count = max(chain_mask)

    return xyz, ss, chain_mask

def process_features(features, max_seq=1280):
    interacting_mask = np.zeros(max_seq)# TODO: 0 is None, 1 is binding substrate, 2 is binding cofactor, 3 is Active residue, proton donater
    tm_mask = np.zeros(max_seq)

    features = features.split('~')
    if len(features) > 1:
        for feat in features:
            f_type, start, end, desc = feat.split('_')
            start = int(start) - 1
            end = int(end) - 1
            if f_type == 'Binding':
                interacting_mask[start:end] = 1
            elif f_type == 'Active':
                interacting_mask[start:end] = 2
            elif f_type == 'TM':
                tm_mask[start:end] = 1
    
    return interacting_mask, tm_mask

def process_cofactors(cofactors, n_cofactors=4):

    cofactors_arr = np.zeros(n_cofactors)
    idx = 0
    for cof in cofactors.split('_'):
        cof = cof.replace('\n', '')
        if cof in cofactors_dict.keys():
            token = cofactors_dict[cof]
            cofactors_arr[idx] = token
        else:
            cofactors_dict[cof] = len(cofactors_dict.keys())
            with open('datasets/cofactors.json', 'w+') as f:
                json.dump(cofactors_dict, f)
    return cofactors_arr

if __name__ == '__main__':
    # p = parse_tsv('datasets/raw_enzyme_data.tsv', 'datasets/enzyme_data')
    p = parse_tsv('datasets/test.tsv', 'datasets/test_v2')
