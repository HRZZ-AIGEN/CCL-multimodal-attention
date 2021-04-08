from abc import ABC
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondDir as BD
from pathlib import Path
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, Batch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from torch.utils.data import Dataset


class MolData(Data):
    def __cat_dim__(self, key, item):
        if key == 'x' or key == 'dist_matrix' or key == 'adj_matrix':
            return None
        else:
            return super().__cat_dim__(key, item)


class MolecularGraphs(InMemoryDataset, ABC):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
              BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
    direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                 BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                 BD.UNKNOWN: 6}

    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return Path(self.root)

    @property
    def processed_dir(self):
        return Path(self.root) / 'pytorch_graphs/'

    @property
    def raw_file_names(self):
        return "labeled_data.csv"

    @property
    def processed_file_names(self):
        return "molecular_graphs.pt"

    def process(self):
        """ Load data """
        bad_cids = [117560, 23939, 0, 23976]  # don't save these molecules

        labels = pd.read_csv(self.raw_paths[0], index_col=0)
        labels = labels.loc[~labels['pubchem_cid'].isin(bad_cids)]
        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

        molecules = list(labels['pubchem_cid'].unique())
        molecule_list = []
        names = []
        data_list = []

        print("Creating molecular graphs")
        for molecule in enumerate(tqdm((molecules),
                                       total=len(molecules),
                                       position=0,
                                       leave=True)):
            name = molecule[1]
            smiles = labels.loc[labels['pubchem_cid'] == name]['smiles']
            mol = Chem.MolFromSmiles(smiles.values[0])  # there are multiple returned values

            """ Features """
            atomic_number = []
            aromatic = []
            donor = []
            acceptor = []
            s = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []

            for atom in mol.GetAtoms():
                # type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                donor.append(0)
                acceptor.append(0)
                s.append(1 if hybridization == HybridizationType.S
                         else 0)
                sp.append(1 if hybridization == HybridizationType.SP
                          else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2
                           else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3
                           else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D
                            else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                             else 0)

                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x = torch.tensor([atomic_number,
                              acceptor,
                              donor,
                              aromatic,
                              s, sp, sp2, sp3, sp3d, sp3d2,
                              num_hs],
                             dtype=torch.float).t().contiguous()

            row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                bond_stereo += 2 * [self.stereo[bond.GetStereo()]]
                bond_dir += 2 * [self.direction[bond.GetBondDir()]]
                # 2* list, because the bonds are defined 2 times, start -> end,
                # and end -> start
            edge_index = torch.tensor([row, col], dtype=torch.long)

            """ Create distance matrix """
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            conf = mol.GetConformer()
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())])
            dist_matrix = pairwise_distances(pos_matrix)

            adj_matrix = np.eye(mol.GetNumAtoms())
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom().GetIdx()
                end_atom = bond.GetEndAtom().GetIdx()
                adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

            keys = ['x', 'name', 'distance_matrix', 'adj_matrix', 'edge_index']
            values = [x, name, dist_matrix, adj_matrix, edge_index]
            molecule_dict = dict(zip(keys, values))
            molecule_list.append(molecule_dict)
            names.append(name)

        print("Concatenating the dataset")
        for molecule_data in molecule_list:
            name = molecule_data.get("name")
            x = molecule_data.get("x")
            distance_matrix = molecule_data.get("distance_matrix")
            adj_matrix = molecule_data.get("adj_matrix")

            data = MolData(
                x=x,
                drug_name=name,
                adj_matrix=adj_matrix,
                dist_matrix=distance_matrix,
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class PPIGraphs(InMemoryDataset, ABC):

    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return Path(self.root)

    @property
    def processed_dir(self):
        return Path(self.root) / 'pytorch_graphs/'

    @property
    def raw_file_names(self):
        return "labeled_data.csv"

    @property
    def processed_file_names(self):
        return "ppi_graphs.pt"

    def process(self):
        print("Creating PPI graphs")

        data_list = []

        project_dir = Path(self.root).resolve().parents[0].absolute()
        mutations = pd.read_csv(project_dir / 'raw/depmap20q2/CCLE_mutations.csv', sep='\t')
        expression = pd.read_csv(project_dir / 'raw/depmap20q2/CCLE_expression.csv', index_col=0)
        landmark_genes = pd.read_csv(project_dir / 'raw/landmark_genes_l1000.txt', sep='\t')
        sample_info = pd.read_csv(project_dir / 'raw/depmap20q2/sample_info.csv')
        ppi_links = pd.read_csv(project_dir / 'raw/9606.protein.links.full.v11.0.txt', delim_whitespace=True)
        ppi_info = pd.read_csv(project_dir / 'raw/9606.protein.info.v11.0.txt', sep='\t')
        cnv = pd.read_csv(project_dir / 'raw/depmap20q2/CCLE_gene_cn.csv', index_col=0)

        # edit expression data
        landmark_genes_list = list(landmark_genes['pr_gene_symbol**'])
        expression.columns = expression.columns.str.split(' ').str[0]
        cols = [col for col in expression.columns if col in landmark_genes_list]  # TPM of landmark genes only
        expression = expression[cols]
        expression.reset_index(inplace=True)
        expression.rename(columns={'index': 'DepMap_ID'}, inplace=True)
        expression = expression.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
        expression.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
        expression = pd.melt(expression, id_vars=['cellosaurus_accession', 'DepMap_ID'], var_name='gene',
                             value_name='tpm')
        expression.dropna(inplace=True)

        # edit copy number variation data
        cnv.columns = cnv.columns.str.split(' ').str[0]
        cnv_cols = [col for col in cnv.columns if col in landmark_genes_list]
        cnv = cnv[cnv_cols]
        cnv.reset_index(inplace=True)
        cnv.rename(columns={'index': 'DepMap_ID'}, inplace=True)
        cnv = cnv.merge(sample_info[['DepMap_ID', 'RRID']], how='inner', on='DepMap_ID')
        cnv.rename(columns={'RRID': 'cellosaurus_accession'}, inplace=True)
        cnv = pd.melt(cnv, id_vars=['cellosaurus_accession', 'DepMap_ID'], var_name='gene',
                      value_name='cnv')
        cnv.dropna(inplace=True)

        expression = expression.merge(cnv[['cellosaurus_accession', 'gene', 'cnv']], how='inner',
                                      on=['cellosaurus_accession', 'gene'])

        # edit PPI network data
        ppi_links = ppi_links.loc[ppi_links['experiments'] != 0][['protein1', 'protein2', 'combined_score']]
        ppi_links = ppi_links.merge(ppi_info[['protein_external_id', 'preferred_name']], how='left',
                                    left_on='protein1', right_on='protein_external_id')
        ppi_links = ppi_links.merge(ppi_info[['protein_external_id', 'preferred_name']], how='left',
                                    left_on='protein2', right_on='protein_external_id')
        ppi_links.drop(columns=['protein_external_id_x', 'protein_external_id_y', 'protein1', 'protein2'],
                       inplace=True)

        """ further reduce the dimensionality of the dataset"""
        ppi_links.rename(columns={
            'preferred_name_x': 'protein_1',
            'preferred_name_y': 'protein_2',
        }, inplace=True)

        # edit mutation data
        mutations.rename(columns={'Hugo_Symbol': 'gene'}, inplace=True)
        expression = expression.merge(mutations[['gene', 'DepMap_ID', 'Variant_Classification', 'Variant_Type']],
                                      how='left',
                                      left_on=['DepMap_ID', 'gene'],
                                      right_on=['DepMap_ID', 'gene'])
        expression['Variant_Classification'] = expression['Variant_Classification'].fillna(value='Wild_Type')
        expression['Variant_Type'] = expression['Variant_Type'].fillna(value='WT')

        expression.drop_duplicates(subset=['cellosaurus_accession', 'gene'], inplace=True)
        variant_type_encoder = OneHotEncoder(sparse=False)
        variant_type_oh = variant_type_encoder.fit_transform(expression[['Variant_Type']])
        expression['variant_type_oh'] = variant_type_oh.tolist()
        ppi_links_cell = ppi_links.loc[(ppi_links['protein_1'].isin(expression['gene'])) &
                                       (ppi_links['protein_2'].isin(expression['gene']))]
        expression.to_csv(project_dir / 'processed/cell_features.csv')
        ppi_links_cell.to_csv(project_dir / 'processed/ppi_links.csv')

        print("Creating PPI graphs")
        cells = list(expression['cellosaurus_accession'].unique())
        for cell in tqdm(cells, position=0, leave=True):
            cell_expression = expression.loc[expression['cellosaurus_accession'] == cell]
            graph = nx.from_pandas_edgelist(ppi_links_cell, source='protein_1', target='protein_2')
            nx.set_node_attributes(graph, pd.Series(cell_expression.tpm.values, index=cell_expression.gene).to_dict(),
                                   'tpm')
            nx.set_node_attributes(graph, pd.Series(cell_expression.variant_type_oh.values,
                                                    index=cell_expression.gene).to_dict(), 'variant_type')
            nx.set_node_attributes(graph, pd.Series(cell_expression['gene'].values,
                                                    index=cell_expression.gene).to_dict(), 'node_name')
            nx.set_node_attributes(graph,
                                   pd.Series(cell_expression.cnv.values, index=cell_expression.gene).to_dict(),
                                   'cnv')

            pytorch_graph = from_networkx(graph)
            x_ppi = torch.cat([pytorch_graph['tpm'].unsqueeze(-1),
                               pytorch_graph['cnv'].unsqueeze(-1),
                               pytorch_graph['variant_type'],
                               ],
                              dim=-1).to(torch.float)
            ppi_edge_index = pytorch_graph['edge_index']
            data = Data(x=x_ppi,
                        edge_index=ppi_edge_index,
                        cell_name=cell,
                        node_name=pytorch_graph['node_name'])
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path):
        self.root = Path(__file__).resolve().parents[1].absolute()
        self.molecular_graphs = MolecularGraphs(self.root / "data/processed")
        self.ppi_graphs = PPIGraphs(self.root / "data/processed")
        self.labels = pd.read_csv(labels_path, index_col=0)[['pubchem_cid', 'cellosaurus_accession', 'sensitivity_uM']]
        self.targets = self.labels['sensitivity_uM']

        # get indices with pubchem cids
        mol_index = []
        pubchem_cid = []
        i = 0
        for graph in self.molecular_graphs:
            mol_index.append(i)
            pubchem_cid.append(graph.drug_name)
            i += 1
        self.mol_indices = pd.Series(data=mol_index, index=pubchem_cid)

        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, index=cellosaurus_accession)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        mol_graph_idx = self.mol_indices[(item['pubchem_cid'])]
        ppi_graph_idx = self.cell_indices[item['cellosaurus_accession']]
        return self.molecular_graphs[mol_graph_idx.tolist()], self.ppi_graphs[ppi_graph_idx.tolist()], item[
            'sensitivity_uM']

    def __len__(self):
        return len(self.labels)

    def balance_sampler(self):
        sensitive = self.labels.sensitivity_uM.value_counts()[1]
        resistant = self.labels.sensitivity_uM.value_counts()[0]
        return [resistant, sensitive]


def collate(data_list):
    """Batch two different types of data together"""
    list_adj_mat = []
    list_dist_mat = []
    list_node_feat = []
    max_shape = max([data[0].dist_matrix.shape[0] for data in data_list])
    for data in data_list:
        list_adj_mat.append(pad_array(data[0].adj_matrix, (max_shape, max_shape)))
        list_dist_mat.append(pad_array(data[0].dist_matrix, (max_shape, max_shape)))
        list_node_feat.append(pad_array(data[0].x, (max_shape, data[0].x.shape[1])))

    batchA = [torch.Tensor(features) for features in (list_adj_mat, list_dist_mat, list_node_feat)]
    batchB = Batch.from_data_list([data[1] for data in data_list])

    try:
        target = torch.Tensor([data[2] for data in data_list])
        return batchA, batchB, target
    except:
        return batchA, batchB

def pad_array(array, shape, dtype=np.float32):
    """Pad 2D input molecular arrays to same size"""
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


"""MolecularGraphs class for benchmarking regression on GDSC dataset"""


class MolecularGraphsGDSC(InMemoryDataset, ABC):
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
              BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
    direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                 BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                 BD.UNKNOWN: 6}

    def __init__(self, root):
        super().__init__(root)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = Path(root)

    @property
    def raw_dir(self):
        return Path(self.root)

    @property
    def processed_dir(self):
        return Path(self.root / 'pytorch_graphs/')

    @property
    def raw_file_names(self):
        return "gdsc_labeled_data.csv"

    @property
    def processed_file_names(self):
        return "gdsc_molecular_graphs.pt"

    def process(self):
        """ Load data """
        labels = pd.read_csv(self.raw_paths[0], index_col=0)
        fdef_name = Path(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

        molecules = list(labels['drug_name'].unique())
        molecule_list = []
        names = []
        data_list = []

        print("Creating molecular graphs")
        for molecule in enumerate(tqdm((molecules),
                                       total=len(molecules),
                                       position=0,
                                       leave=True)):
            name = molecule[1]
            smiles = labels.loc[labels['drug_name'] ==
                                name]['smiles']
            mol = Chem.MolFromSmiles(smiles.values[0])  # there are multiple returned values
            N = mol.GetNumAtoms()

            """ Features """
            atomic_number = []
            aromatic = []
            donor = []
            acceptor = []
            s = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []

            for atom in mol.GetAtoms():
                # type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                donor.append(0)
                acceptor.append(0)
                s.append(1 if hybridization == HybridizationType.S
                         else 0)
                sp.append(1 if hybridization == HybridizationType.SP
                          else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2
                           else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3
                           else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D
                            else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                             else 0)

                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x = torch.tensor([atomic_number,
                              acceptor,
                              donor,
                              aromatic,
                              s, sp, sp2, sp3, sp3d, sp3d2,
                              num_hs],
                             dtype=torch.float).t().contiguous()

            row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                bond_stereo += 2 * [self.stereo[bond.GetStereo()]]
                bond_dir += 2 * [self.direction[bond.GetBondDir()]]
                # 2* list, because the bonds are defined 2 times, start -> end,
                # and end -> start
            edge_index = torch.tensor([row, col], dtype=torch.long)

            """ Create distance matrix """

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=200000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

            # AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())])
            dist_matrix = pairwise_distances(pos_matrix)

            adj_matrix = np.eye(mol.GetNumAtoms())
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom().GetIdx()
                end_atom = bond.GetEndAtom().GetIdx()
                adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

            keys = ['x', 'name', 'distance_matrix', 'adj_matrix', 'edge_index']
            values = [x, name, dist_matrix, adj_matrix, edge_index]
            molecule_dict = dict(zip(keys, values))
            molecule_list.append(molecule_dict)
            names.append(name)

        print("Concatenating the dataset")
        for molecule_data in molecule_list:
            name = molecule_data.get("name")
            x = molecule_data.get("x")
            distance_matrix = molecule_data.get("distance_matrix")
            adj_matrix = molecule_data.get("adj_matrix")

            data = MolData(
                x=x,
                drug_name=name,
                adj_matrix=adj_matrix,
                dist_matrix=distance_matrix,
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class PairDatasetBenchmark(torch.utils.data.Dataset):
    def __init__(self, labels_path, setting='classification'):
        self.root = Path(__file__).resolve().parents[1].absolute()
        self.molecular_graphs = MolecularGraphsGDSC(self.root / "data/processed")
        self.ppi_graphs = PPIGraphs(self.root / "data/processed")
        self.labels = pd.read_csv(labels_path, index_col=0)[['drug_name', 'cellosaurus_accession', 'pIC50']]

        # get indices with pubchem cids
        mol_index = []
        pubchem_cid = []
        i = 0
        for graph in self.molecular_graphs:
            mol_index.append(i)
            pubchem_cid.append(graph.drug_name)
            i += 1
        self.mol_indices = pd.Series(data=mol_index, index=pubchem_cid)

        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, index=cellosaurus_accession)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        mol_graph_idx = self.mol_indices[(item['drug_name'])]
        ppi_graph_idx = self.cell_indices[item['cellosaurus_accession']]
        return self.molecular_graphs[mol_graph_idx.tolist()], self.ppi_graphs[ppi_graph_idx.tolist()], item[
            'pIC50']

    def __len__(self):
        return len(self.labels)



"""Dataset creator for creating datasets for inference"""


class InferenceDataset(torch.utils.data.Dataset):
    """PPI graphs are read from the torch PPI graphs file, while mol graphs are created"""
    def __init__(self, drug_id, smiles, cellosaurus_accession):
        self.labels = (pd.DataFrame({'smiles': smiles,
                                     'cellosaurus_accession': cellosaurus_accession,
                                     'drug_id': drug_id}))
        self.root = Path(__file__).resolve().parents[1].absolute()
        self.ppi_graphs = PPIGraphs(self.root/'data/processed/')

        # get indices with cellosaurus accessions
        cell_index = []
        cellosaurus_accession = []
        i = 0
        for ppi_graph in self.ppi_graphs:
            cell_index.append(i)
            cellosaurus_accession.append(ppi_graph.cell_name)
            i += 1
        self.cell_indices = pd.Series(data=cell_index, index=cellosaurus_accession)

        self.molecular_graphs = []
        mol_index = []
        drug_ids = []
        i = 0
        for index, mol_graph in self.labels.iterrows():
            self.molecular_graphs.append(self.make_mol_graph(mol_graph['smiles']))
            drug_ids.append(mol_graph['drug_id'])
            mol_index.append(i)
            i += 1
        self.molecular_indices = pd.Series(data=mol_index, index=drug_ids)

    def __getitem__(self, idx):
        item = self.labels.iloc[idx]
        ppi_graph_idx = self.cell_indices[item['cellosaurus_accession']]
        mol_graph_idx = self.molecular_indices[item['drug_id']]
        return self.molecular_graphs[mol_graph_idx.tolist()], self.ppi_graphs[ppi_graph_idx.tolist()]

    def __len__(self):
        return len(self.labels)

    def make_mol_graph(self, smiles):
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
                  BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
        direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                     BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                     BD.UNKNOWN: 6}

        fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
        mol = Chem.MolFromSmiles(smiles)
        N = mol.GetNumAtoms()

        """ Features """
        atomic_number = []
        aromatic = []
        donor = []
        acceptor = []
        s = []
        sp = []
        sp2 = []
        sp3 = []
        sp3d = []
        sp3d2 = []
        num_hs = []

        for atom in mol.GetAtoms():
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            donor.append(0)
            acceptor.append(0)
            s.append(1 if hybridization == HybridizationType.S
                     else 0)
            sp.append(1 if hybridization == HybridizationType.SP
                      else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2
                       else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3
                       else 0)
            sp3d.append(1 if hybridization == HybridizationType.SP3D
                        else 0)
            sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                         else 0)

            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1

        x = torch.tensor([atomic_number,
                          acceptor,
                          donor,
                          aromatic,
                          s, sp, sp2, sp3, sp3d, sp3d2,
                          num_hs],
                         dtype=torch.float).t().contiguous()

        row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [bonds[bond.GetBondType()]]
            bond_stereo += 2 * [stereo[bond.GetStereo()]]
            bond_dir += 2 * [direction[bond.GetBondDir()]]
            # 2* list, because the bonds are defined 2 times, start -> end,
            # and end -> start

        """ Create distance matrix """
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            AllChem.Compute2DCoords(mol)

        conf = mol.GetConformer()
        pos_matrix = np.array(
            [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
             for k in range(mol.GetNumAtoms())])
        dist_matrix = pairwise_distances(pos_matrix)

        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

        data = MolData(
            x=x,
            adj_matrix=adj_matrix,
            dist_matrix=dist_matrix,
        )

        return data









