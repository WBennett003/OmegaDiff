import requests
import re
from requests.adapters import HTTPAdapter, Retry
import os
import time
from queue import Queue
from threading import Thread

from utils import process_mols, AminoAcid_tokenise

"""
    Unirpot returned field docs - https://www.uniprot.org/help/return_fields
    Uniprot quiry docs - https://www.uniprot.org/help/api_queries
"""

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)

def process_html_equation(html_equation, equation):
    chebi_equation = ''

    half_split = equation.split(' = ')
    reactants = half_split[0].split(' + ')
    products = half_split[1].split(' + ')
    n_reactants = len(reactants)
    n_products = len(products)

    splices = html_equation.split('molid=\"chebi:')
    reactant_ids = []
    product_ids = []

    for i, cut in enumerate(splices[1:]):
        idx = cut.index('"') #the first char after the id is / but this causes an error so i will -1 from the position of "
        id = cut[:idx]
        
        if i == n_reactants - 1:
            chebi_equation += id + ' = '
        elif i == len(splices) - 2:
            chebi_equation += id
        else:
            chebi_equation += id + ' + '
        
        
        if i < n_reactants:
            reactant_ids.append(id)
        else:
            product_ids.append(id)
    
    #convert to string to make sample data tabular
    # reactant_ids = ' '.join(reactant_ids)
    # product_ids = ' '.join(product_ids)


    return reactant_ids, product_ids, chebi_equation

def get_rhea(rhea_id, url='https://www.rhea-db.org/rhea/?query='):
    try:
        data = dict(requests.get(url+rhea_id[5:]+'&format=json').json())['results']
        if len(data) > 0:
            data = data[0]
            equation = data['equation']
            html_equation = data['htmlequation']
            reactants, products, chebi_equation = process_html_equation(html_equation, equation)
            return equation, chebi_equation, reactants, products
        return '', '', '', ''
    except:
        return '', '', '', ''

def get_pdb(pdb_code, url='https://files.rcsb.org/download/', dir='datasets/pdb/'):
    if not os.path.isfile(dir+pdb_code+'.pdb'):
        if len(pdb_code) != 4:
            url = url+pdb_code+'.pdb'
        else:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb"
        res = requests.get(url+pdb_code+'.pdb').text
        if res[0] == '<':#got http file
            return
        with open(dir+pdb_code+'.pdb', 'w+') as f:
            f.write(res)

class dataset():
    def __init__(self, 
    max_mol_size = 1000, max_aa_seq = 5000, file_dir = 'datasets/raw_enzyme_data', max_samples=5000000, checkpoint=True, workers=20000):
        self.max_mol_size = max_mol_size
        self.max_aa_seq = max_aa_seq        
        self.file_dir = file_dir
        # self.starting_urls = get_starting_urls(get_batch(self.url), url_depth, workers)

        if not os.path.isfile('checkpoint.txt'):
            self.url = 'https://rest.uniprot.org/uniprotkb/search?format=json&query=GO:0003824&size=500'
            with open('checkpoint.txt', 'w+') as f:
                f.write(self.url)
        else:
            with open('checkpoint.txt', 'r') as f:
                self.url = f.read()

        self.columns=['accension', 'id', 'sequence', 'length', 'taxId', 'name', 'ecId', 'equation', 'chebi_equation', 'smile_equation', 'reactants', 'products', 'Rhea_id', 'features', 'pdb', 'cofactors']

        if not os.path.isfile(file_dir+'.tsv'):
            with open(file_dir+'.tsv', 'w+') as f:
                f.write('\t'.join(self.columns)+'\n')

        queue = Queue()
        for w in range(workers):
            worker = daemon(queue, id=w)

            worker.daemon = True
            worker.start()


        for batch, total in get_batch(self.url):
            self.save_chkpoint(batch)
            queue.put(batch)

        queue.join()

    def save_chkpoint(self, url):
        with open('checkpoint.txt', 'w+') as f:
            f.write(url.url)


class daemon(Thread):
    def __init__(self, queue, id=0):
        Thread.__init__(self)
        self.queue = queue
        self.id = id

    def run(self):
        url = self.queue.get()
        try:
            get_enzyme(url, self.id)
        finally:
            self.queue.task_done()

def get_enzyme(data, worker_id):
    data = dict(data.json())['results']
    count = 0
    for sample in data:
        accension = sample['primaryAccession']
        uniprot_id = sample['uniProtkbId']
        sequence = sample['sequence']['value']
        seq_len = sample['sequence']['length']
        tax_id = sample['organism']['taxonId']
        try:
            name = sample['proteinDescription']['recommendedName']['fullName']['value']
        except:
            name = 'N/A'

        equation, chebi_equation, reactants, products = '', '', '', '' #to check if cross ref is present
        if 'comments' in sample.keys():
            cata = []
            cfs = []
            for comms in sample['comments']:
                if comms['commentType'] == 'CATALYTIC ACTIVITY':
                    if 'ecNumber' in comms['reaction'].keys():
                        ec_id = comms['reaction']['ecNumber']
                    elif 'ecNumbers' in sample['proteinDescription']['recommendedName'].keys():
                        ec_id = sample['proteinDescription']['recommendedName']['ecNumbers'][0]['value']
                    else:
                        ec_id = '0.0.0.0'
                    if 'reactionCrossReferences' in comms['reaction'].keys():
                        for db in comms['reaction']['reactionCrossReferences']:
                            if db['database'] == 'Rhea':
                                Rhea_id = db['id']
                                equation, chebi_equation, reactants, products = get_rhea(Rhea_id)
                                reactants_smiles, products_smiles, smile_reaction = process_mols(reactants, products)
                                if reactants_smiles != '':
                                    cata.append(
                                        [ec_id, Rhea_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles]
                                    )

                elif comms['commentType'] == 'COFACTOR':
                    if 'cofactors' not in comms.keys():
                        print(comms)
                    else:

                        for cofactor in comms['cofactors']:
                            id = cofactor['cofactorCrossReference']['id']
                            cfs.append(id)
            cfs = '_'.join(cfs)

            if 'features' in sample.keys():
                feat = []
                for feature in sample['features']:
                    if feature['type'] == 'Active site':
                        start = str(feature["location"]['start']['value'])
                        end = str(feature["location"]['end']['value'])
                        des = feature['description']
                        feat.append(
                            '_'.join(['Active', start, end, des])
                        )

                    elif feature['type'] == 'Binding site':
                        start = str(feature["location"]['start']['value'])
                        end = str(feature["location"]['end']['value'])
                        ligand = feature['ligand']['name']
                        feat.append(
                            '_'.join(['Binding', start, end, ligand])
                        )

                    elif feature['type'] == 'Transmembrane':
                        start = str(feature["location"]['start']['value'])
                        end = str(feature["location"]['end']['value'])
                        ss = feature['description']
                        feat.append(
                            '_'.join(['TM', start, end, ss])
                        )

                    elif feature['type'] == 'Modified residue':
                        start = str(feature["location"]['start']['value'])
                        end = str(feature["location"]['end']['value'])
                        res = feature['description']
                        feat.append(
                            '_'.join(['Mod', start, end, res])
                        )

                feat = '~'.join(feat)
                pdb_code = []
                alphafold_code = []
                for db in sample['uniProtKBCrossReferences']:
                    if db['database'] == 'PDB':
                        pdb_code.append(db['id'])
                    elif db['database'] == 'AlphaFoldDB':
                        alphafold_code.append(db['id'])
                if len(pdb_code) > 0:
                    pdb_code = pdb_code[0]
                elif pdb_code == [] and alphafold_code != []:
                    pdb_code = alphafold_code[0]
                else:
                    pdb_code = ''


            for c in cata:
                ec_id, Rhea_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles = c
                row = [
                    accension, uniprot_id, sequence, str(seq_len), str(tax_id), name, ec_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles, Rhea_id, feat, pdb_code, cfs
                ]
                row = '\t'.join(row)
                # with open('datasets/shotgun/uniprot_worker'+str(worker_id)+'.tsv', 'a') as f:
                with open('datasets/raw_enzyme_data.tsv', 'a') as f:
                    f.write(row+'\n')
                count += 1
            get_pdb(pdb_code)
    return count



if __name__ == '__main__':
    dataset()