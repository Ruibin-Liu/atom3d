import numpy as np
import pandas as pd
import collections as col
import logging
import os
import re
import sys
import scipy.spatial
import parallel as par
import click

sys.path.insert(0, '../../..')
import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo
import util as res_util


logger = logging.getLogger(__name__)

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C', 'O']
# allowed_atoms = ['C', 'O', 'N', 'S', 'P', 'SE']  # commented on because we only have proteins
allowed_atoms = ['C', 'O', 'N', 'S']

# computed statistics from training set; since 'CYS' is the least common residue and its weight is just 1.0, we don't need to change this
res_wt_dict = {'HIS': 0.581391659111514, 'LYS': 0.266061611865989, 'ARG': 0.2796785729861747, 'ASP': 0.26563454667840314, 'GLU': 0.22814679094919596, 'SER': 0.2612916369563003, 'THR': 0.27832512315270935, 'ASN': 0.3477441570413752, 'GLN': 0.37781509139381086, 'ALA': 0.20421144813311043, 'VAL': 0.22354397064847012, 'LEU': 0.18395198072344454, 'ILE': 0.2631600545792168, 'MET': 0.6918305148744505, 'PHE': 0.3592224851905275, 'TYR': 0.4048964515721682, 'TRP': 0.9882874205355423, 'PRO': 0.32994186046511625, 'GLY': 0.2238561093317741, 'CYS': 1.0}

gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ], dtype=np.float32)
# gly_CB_sigma = np.array([[1.63731114e-03, 2.40018381e-04, 6.38361679e-04],
#        [2.40018381e-04, 6.87853419e-05, 1.43898267e-04],
#        [6.38361679e-04, 1.43898267e-04, 3.25022011e-04]], dtype=np.float32)


class LigCysTransform(object):

    def __init__(self, pos_residues=None, neg_residues=None, balance=False):
        self.balance = balance
        if pos_residues:
            with open(pos_residues, 'r') as f:
                self.pos_residues = set([tuple(x.strip().split(',')) for x in f.readlines()])
        else:
            self.pos_residues = set()
        if neg_residues:
            with open(neg_residues, 'r') as f:
                self.neg_residues = set([tuple(x.strip().split(',')) for x in f.readlines()])
        else:
            self.neg_residues = set()
    
    def __call__(self, x):
        x['id'] = fi.get_pdb_code(x['id'])
        df = x['atoms']

        subunits = []
        df = df.dropna(subset=['x','y','z'])
        #remove Hets and non-allowable atoms
        df = df[df['element'].isin(allowed_atoms)]
        df = df[df['hetero'].str.strip()=='']
        df = df.reset_index(drop=True)
        
        , input_file_path=Nonelabels = []
        for chain_res, res_df in df.groupby(['structure', 'chain', 'residue', 'resname']):
            structure, chain, res, res_name = chain_res
            target_resid = structure.split('/')[-1].rstrip('.pdb').split('_')[-2] # atom3d_pdbs/A0FGR8-1_C181_78_022.pdb: 78
            # only train on CYS
            if res_name != 'CYS' or str(res) != target_resid:
                continue
            # sample each residue based on its frequency in train data
            if self.balance:
                if not np.random.random() < res_wt_dict[res_name]:
                    continue

            if not np.all([b in res_df['name'].to_list() for b in bb_atoms]):
                continue
            CB_pos = res_df[res_df['name']=='CB'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

            subunit_df = df
            
            # environment = all atoms within 10*sqrt(3) angstroms (to enable a 20A cube)
            kd_tree = scipy.spatial.KDTree(subunit_df[['x','y','z']].to_numpy())
            subunit_pt_idx = kd_tree.query_ball_point(CB_pos, r=10.0*np.sqrt(3), p=2.0)
            label = -1
            if (f'atom3d_pdbs/{structure}', chain, str(res), res_name) in self.pos_residues:
                label = 1
            elif (f'atom3d_pdbs/{structure}', chain, str(res), res_name) in self.neg_residues:
                label = 0
            else:
                continue
             
            subunits.append(subunit_df.index[sorted(subunit_pt_idx)].to_list())
            label_row = [structure[:-4], label, CB_pos[0], CB_pos[1], CB_pos[2]]
            labels.append(label_row)
            break
        assert len(labels) == len(subunits)
        x['atoms'] = df
        x['labels'] = pd.DataFrame(labels, columns=['subunit', 'label', 'x', 'y', 'z'])
        x['subunit_indices'] = subunits

        return x

def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir, cv_index=None):
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])
        # Check if the target in id is in the desired target split set
        split_ids = list(filter(lambda id: id in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join([str(i) for i in split_indices])))
        return split_indices

    logger.info(f'Write results to {split_dir:}...')
    os.makedirs(os.path.join(split_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'data'), exist_ok=True)
    train_index_file = 'indices/train_indices.txt'
    train_db_file = 'data/train'
    val_index_file = 'indices/val_indices.txt'
    val_db_file = 'data/val'
    if cv_index is not None:
        train_index_file = f'indices/train{cv_index}_indices.txt'
        val_index_file = f'indices/val{cv_index}_indices.txt'
        train_db_file = f'data/train{cv_index}'
        val_db_file = f'data/val{cv_index}'
    indices_train = _write_split_indices(
        train_txt, lmdb_ds, os.path.join(split_dir, train_index_file))
    indices_val = _write_split_indices(
        val_txt, lmdb_ds, os.path.join(split_dir, val_index_file))
    indices_test = _write_split_indices(
        test_txt, lmdb_ds, os.path.join(split_dir, 'indices/test_indices.txt'))

    train_dataset, val_dataset, test_dataset = spl.split(
        lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_dir, train_db_file))
    da.make_lmdb_dataset(val_dataset, os.path.join(split_dir, val_db_file))
    if cv_index is None or cv_index == 0:
        da.make_lmdb_dataset(test_dataset, os.path.join(split_dir, 'data/test'))

@click.command(help='Prepare LigCys dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--split', '-s', is_flag=True)
@click.option('--balance', '-b', is_flag=True)
@click.option('--pos_residues_csv', '-p', type=click.Path(exists=True), default=None)
@click.option('--neg_residues_csv', '-n', type=click.Path(exists=True), default=None)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
@click.option('--cv_fold', '-cv', type=int, default=1)
@click.option('--num_threads', '-nt', type=int, default=8)
@click.option('--start', '-st', type=int, default=0)
def prepare(input_file_path, output_root, split, balance, pos_residues_csv, neg_residues_csv, train_txt, val_txt, test_txt, cv_fold, num_threads, start):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                       level=logging.INFO)
    
    def _process_chunk(file_list, filetype, lmdb_path, balance):
        logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
        if not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        dataset = da.load_dataset(
            file_list, filetype, 
            transform=LigCysTransform(balance=balance, pos_residues=pos_residues_csv, neg_residues=neg_residues_csv))
        da.make_lmdb_dataset(dataset, lmdb_path)

    # Assume PDB filetype.
    filetype = 'pdb'

    file_list = fi.find_files(input_file_path, fo.patterns[filetype])
    
    lmdb_path = os.path.join(output_root, 'raw/LigCys/data')
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    
    # Comment out below two lines to download by chunks
    #######
    dataset = da.load_dataset(
        file_list, filetype,
        transform=ABPPTransform(balance=balance, pos_residues=pos_residues_csv, neg_residues=neg_residues_csv))
    da.make_lmdb_dataset(dataset, lmdb_path)
    #######
    
    # Comment out below eight lines to download entire dataset (takes much longer)
    #######
    # chunk_size = (len(file_list) // num_threads) + 1
    # chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    # assert len(chunks) == num_threads
    
    # for i in range(start,num_threads):
    #     logger.info(f'Processing chunk {i:}...')
    #     _process_chunk(chunks[i], 'pdb', f'{lmdb_path}_tmp_{i}', balance)
        
    # cmd = f'python combine_lmdb.py {lmdb_path}_tmp_* {lmdb_path}_all'
    # os.system(cmd)
    #######
        
    if split:
        output_root = os.path.join(output_root, 'splits/split-by-clustering')
        if cv_fold == 1: 
            split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, output_root)
        elif cv_fold > 1:
            for cv in range(cv_fold):
                path_base = test_txt.split('/')
                train_txt = '/'.join(path_base[:-1]) + f'/train{cv}_ids.txt'
                val_txt = '/'.join(path_base[:-1]) + f'/val{cv}_ids.txt'
                split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, output_root, cv_index=cv)


if __name__ == "__main__":
    prepare()
    print('Done!')
