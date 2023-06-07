import h5py
from tokeniser import Amino_Acid_Tokeniser
import numpy as np
import time
from drfp import DrfpEncoder


def padding(seq, maxlength, padding_idx=21):
    blank = np.zeros(maxlength)
    blank[len(seq):] = padding_idx
    blank[:len(seq)] = seq
    return blank

class parse_tsv:
    def __init__(self, inp_file_dir, out_file_dir, max_seq_aa=1280, reaction_embed_size=10240, max=1000000, hydrogens=True, root_atom=True) -> None:
        self.inp_file_dir = inp_file_dir
        self.out_file_dir = out_file_dir
        self.max_aa_seq = max_seq_aa
        self.reaction_size = reaction_embed_size
        self.reaction_encoder = DrfpEncoder
        self.hydrogens = hydrogens
        self.root_atom = root_atom

        self.tokeniser = Amino_Acid_Tokeniser()

        self.initalise_ds()

        self.run(max=max)

    def step_row(self, row):
        #clean row
        row = row.split('\t')
        if len(self.columns) == len(row): # if there is an an error with the row
            acc, ID, seq, length, taxid, name, ecid, eqx, cheqx, smilerxn, reactsmile, prodsmile, rhea = row
            if int(length) < self.max_aa_seq: # if the sequence is longer than the standard length
                tokens = self.tokeniser(seq)
                tokens = padding(tokens, self.max_aa_seq)
                if min(tokens) >= 0.0:
                    reaction = self.reaction_encoder.encode(smilerxn, n_folded_length=self.reaction_size, radius=3, include_hydrogens=self.hydrogens, root_central_atom=self.root_atom)
                    self.update_ds(acc, ID, seq, length, taxid, name, ecid, eqx, cheqx, smilerxn, reactsmile, prodsmile, rhea, tokens, reaction)
                else:
                    pass

    def run(self, step=10000, max=None):
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
                    perc = (n - i) / step
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
        self.file.create_dataset('taxId', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('name', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('ecId', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('chebi_equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('smile_equation', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('reactants', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('products', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))        
        
        self.file.create_dataset('reactants_smile', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('products_smile', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))

        self.file.create_dataset('Rhea_id', (0, 1), dtype=string_dt, chunks=True, maxshape=(None, 1))
        self.file.create_dataset('AA_seq', (0, self.max_aa_seq), dtype='i', chunks=True, maxshape=(None, self.max_aa_seq))
        self.file.create_dataset('reaction', (0, self.reaction_size), dtype='i', chunks=True, maxshape=(None, self.reaction_size))

    def update_ds(self, accension, uniprot_id, sequence, seq_len, tax_id, name, ec_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles, Rhea_id, tokenised_sequence, reaction):
        self.save_dataset('accension', accension)
        self.save_dataset('id', uniprot_id)
        self.save_dataset('sequence', sequence)
        self.save_dataset('length', int(seq_len))
        self.save_dataset('taxId', str(tax_id))
        self.save_dataset('name', name)
        self.save_dataset('ecId', ec_id)
        self.save_dataset('equation', equation)
        self.save_dataset('chebi_equation', chebi_equation)
        self.save_dataset('smile_equation', smile_reaction)
        self.save_dataset('reactants_smile', reactants_smiles)
        self.save_dataset('products_smile', products_smiles)
        self.save_dataset('Rhea_id', Rhea_id)
        self.save_dataset('AA_seq', tokenised_sequence)
        self.save_dataset('reaction', reaction)

    def save_dataset(self, key, arr):
        self.file[key].resize((self.file[key].shape[0] + 1), axis=0)
        self.file[key][-1:] = arr

if __name__ == '__main__':
    p = parse_tsv('datasets/uniprot_tokenised_db.tsv', 'datasets/2056_hash_ds_1M')
