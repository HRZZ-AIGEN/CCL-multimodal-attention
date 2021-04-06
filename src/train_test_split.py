from pathlib import Path
import pandas as pd
import random
from create_graphs import PPIGraphs, MolecularGraphsGDSC, MolecularGraphs
import requests
import zipfile
import shutil

def create_gdsc_split(split='random'):
    """Creates a random split, used in benchmarks"""
    random.seed(42)
    root = Path(__file__).resolve().parents[1].absolute()

    # downloads data folder if it doesn't exist
    data_dir = (root / 'data')
    if not data_dir.exists():
        print("Downloading data (1.4 GB)")
        url = "https://www.dropbox.com/sh/h4tpjd64ebemo06/AADLEKi0844C0PQbk-X1ajk0a?dl=1"

        with requests.get(url, stream=True) as r:
            with open(root / "data.zip", 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("Data downloaded!")
        data_dir.mkdir()
        with zipfile.ZipFile(root / "data.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)

    print("Creating train-val-test splits")
    if split == 'random':
        save_dir = (root / 'data/processed/gdsc_benchmark')
        if not save_dir.exists():
            save_dir.mkdir()
            ppi_graphs = PPIGraphs(root / 'data/processed')  # load and/or create PPI graphs
            gdsc = pd.read_csv(root / 'data/raw/GDSC_data/PANCANCER_IC.csv')
            smiles = pd.read_csv(root / 'data/raw/GDSC_data/drug_smiles.csv')
            sample_info = pd.read_csv(root / 'data/processed/sample_info.csv')[['COSMICID', 'RRID']]

            cell_names = []
            for i in ppi_graphs:
                cell_names.append(i.cell_name)

            gdsc = gdsc.merge(sample_info, how='inner', left_on='Cosmic sample Id', right_on='COSMICID')
            gdsc = gdsc.loc[gdsc['RRID'].isin(cell_names)]
            gdsc = gdsc.merge(smiles[['name', 'CanonicalSMILES']], how='inner', left_on='Drug name', right_on='name')
            gdsc.drop_duplicates(subset=['Drug name', 'RRID'], inplace=True)
            gdsc = gdsc.loc[~gdsc['CanonicalSMILES'].str.contains('\.')]
            gdsc.rename(columns={'Drug name': 'drug_name',
                                 'CanonicalSMILES': 'smiles',
                                 'IC50': 'pIC50',
                                 'RRID': 'cellosaurus_accession'}, inplace=True)
            if not Path(root / 'data/processed/gdsc_labeled_data.csv').exists():
                gdsc.to_csv(root / 'data/processed/gdsc_labeled_data.csv')

            # save train-test-splits
            test = gdsc.sample(random_state=42, frac=0.1)
            test.to_csv(save_dir / 'test.csv')
            gdsc = gdsc.drop(test.index)
            val = gdsc.sample(random_state=42, frac=0.1)
            val.to_csv(save_dir / 'val.csv')
            train = gdsc.drop(val.index)
            train.to_csv(save_dir / 'train.csv')

    if split == 'blind':
        save_dir = (root / 'data/processed/gdsc_benchmark_blind')
        if not save_dir.exists():
            save_dir.mkdir()
            ppi_graphs = PPIGraphs(root / 'data/processed')  # load and/or create PPI graphs
            gdsc = pd.read_csv(root / 'data/raw/GDSC_data/PANCANCER_IC.csv')
            smiles = pd.read_csv(root / 'data/raw/GDSC_data/drug_smiles.csv')
            sample_info = pd.read_csv(root / 'data/processed/sample_info.csv')[['COSMICID', 'RRID']]

            cell_names = []
            for i in ppi_graphs:
                cell_names.append(i.cell_name)

            gdsc = gdsc.merge(sample_info, how='inner', left_on='Cosmic sample Id', right_on='COSMICID')
            gdsc = gdsc.loc[gdsc['RRID'].isin(cell_names)]
            gdsc = gdsc.merge(smiles[['name', 'CanonicalSMILES']], how='inner', left_on='Drug name', right_on='name')
            gdsc.drop_duplicates(subset=['Drug name', 'RRID'], inplace=True)
            gdsc = gdsc.loc[~gdsc['CanonicalSMILES'].str.contains('\.')]
            gdsc.rename(columns={'Drug name': 'drug_name',
                                 'CanonicalSMILES': 'smiles',
                                 'IC50': 'pIC50',
                                 'RRID': 'cellosaurus_accession'}, inplace=True)
            if not Path(root / 'data/processed/gdsc_labeled_data.csv').exists():
                gdsc.to_csv(root / 'data/processed/gdsc_labeled_data.csv')

            # save train-test-splits
            """Make 3 test set settings concatenated in one:
                1. Blind cells
                2. Blind drugs
                3. Blind cells and drugs"""
            # blind drugs
            drugs = gdsc['drug_name'].unique()
            n_drugs = int(len(drugs) * 0.15)  # number of unique drugs to draw
            unique_drugs = random.sample(list(drugs), n_drugs)
            blind_drugs = gdsc.loc[gdsc['drug_name'].isin(unique_drugs)]
            data = gdsc.drop(blind_drugs.index)

            # double blind
            double_blind_cells = blind_drugs['cellosaurus_accession'].unique()
            n_cells = int(len(double_blind_cells) * 0.1)
            double_blind_cells = random.sample(list(double_blind_cells), n_cells)
            double_blind = blind_drugs.loc[blind_drugs['cellosaurus_accession'].isin(double_blind_cells)]
            data = data.loc[~data['cellosaurus_accession'].isin(double_blind['cellosaurus_accession'].unique())]

            # blind cells
            blind_cells = data['cellosaurus_accession'].unique()
            n_cells = int(len(blind_cells) * 0.1)
            blind_cells = random.sample(list(blind_cells), n_cells)
            blind_cells = data.loc[data['cellosaurus_accession'].isin(blind_cells)]
            data = data.drop(blind_cells.index)
            val = data.sample(random_state=42, frac=0.1)
            data = data.drop(val.index)

            blind_drugs.to_csv(save_dir / 'blind_drugs.csv')
            blind_cells.to_csv(save_dir / 'blind_cells.csv')
            double_blind.to_csv(save_dir / 'double_blind.csv')
            pd.concat([blind_drugs, blind_cells, double_blind]).to_csv(save_dir / 'test.csv')
            val.to_csv(save_dir / 'val.csv')
            data.to_csv(save_dir / 'train.csv')


    MolecularGraphsGDSC(root / 'data/processed/') # if !exists -> create
    return save_dir


def train_test_split(dataset='GDSC', split='random'):
    """Creates different splits, on different datasets from PharmacoGX R package"""
    random.seed(42)
    root = Path(__file__).resolve().parents[1].absolute()

    data_dir = (root / 'data')
    if not data_dir.exists():
        print("Downloading data (1.4 GB)")
        url = "https://www.dropbox.com/sh/h4tpjd64ebemo06/AADLEKi0844C0PQbk-X1ajk0a?dl=1"

        with requests.get(url, stream=True) as r:
            with open(root / "data.zip", 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("Data downloaded!")

        data_dir.mkdir()
        with zipfile.ZipFile(root / "data.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)

    print("Creating train-val-test splits")
    save_dir = (root / 'data/processed/{}_{}'.format(dataset, split))
    if not save_dir.exists():
        save_dir.mkdir()
        ppi_graphs = PPIGraphs(root / 'data/processed/')  # if !exists -> create
        cell_names = []
        for i in ppi_graphs:
            cell_names.append(i.cell_name)
        # MolecularGraphs(root / 'data/processed')  # if !exists -> create
        data = pd.read_csv(root / 'data/processed/labeled_data.csv', index_col=0)
        data = data.loc[data['dataset_name'] == dataset]
        data = data.loc[data['cellosaurus_accession'].isin(cell_names)]

        bad_cids = [117560, 23939, 0, 23976]
        # drop Cr atom from CTRP
        data = data.drop(data.loc[data["pubchem_cid"].isin(bad_cids)].index)

        if split == 'random':
            """Randomly split on unknown drugs"""
            test = data.sample(random_state=42, frac=0.1)
            test.to_csv(save_dir / 'test.csv')
            data = data.drop(test.index)
            val = data.sample(random_state=42, frac=0.1)
            val.to_csv(save_dir / 'val.csv')
            train = data.drop(val.index)
            train.to_csv(save_dir / 'train.csv')

        if split == 'blind':
            """Make 3 test set settings concatenated in one:
                1. Blind cells
                2. Blind drugs
                3. Blind cells and drugs"""
            # blind drugs
            drugs = data['pubchem_cid'].unique()
            n_drugs = int(len(drugs) * 0.15)  # number of unique drugs to draw
            unique_drugs = random.sample(list(drugs), n_drugs)
            blind_drugs = data.loc[data['pubchem_cid'].isin(unique_drugs)]
            data = data.drop(blind_drugs.index)

            # double blind
            double_blind_cells = blind_drugs['cellosaurus_accession'].unique()
            n_cells = int(len(double_blind_cells) * 0.1)
            if dataset == 'NCI 60':
                n_cells = 5
            double_blind_cells = random.sample(list(double_blind_cells), n_cells)
            double_blind = blind_drugs.loc[blind_drugs['cellosaurus_accession'].isin(double_blind_cells)]
            data = data.loc[~data['cellosaurus_accession'].isin(double_blind['cellosaurus_accession'].unique())]

            # blind cells
            blind_cells = data['cellosaurus_accession'].unique()
            n_cells = int(len(blind_cells) * 0.1)
            if dataset == 'NCI 60':
                n_cells = 5
            blind_cells = random.sample(list(blind_cells), n_cells)
            blind_cells = data.loc[data['cellosaurus_accession'].isin(blind_cells)]
            data = data.drop(blind_cells.index)
            val = data.sample(random_state=42, frac=0.1)
            data = data.drop(val.index)

            blind_drugs.to_csv(save_dir / 'blind_drugs.csv')
            blind_cells.to_csv(save_dir / 'blind_cells.csv')
            double_blind.to_csv(save_dir / 'double_blind.csv')
            pd.concat([blind_drugs, blind_cells, double_blind]).to_csv(save_dir / 'test.csv')
            val.to_csv(save_dir / 'val.csv')
            data.to_csv(save_dir / 'train.csv')

    return save_dir
