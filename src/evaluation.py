import pandas as pd
import torch
from torch.utils.data import DataLoader
from create_graphs import PairDataset, collate, InferenceDataset, PPIGraphs
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from train_classification_models import MultimodalAttentionNet, Conf
from sklearn.metrics import average_precision_score, roc_auc_score

conf = Conf(
    lr=1e-4,
    batch_size=32,
    epochs=300,
    reduce_lr=True
).to_hparams()

class Evaluation:
    def __init__(self, root_dir, dataset='GDSC', split='random', ppi_depth=3, seed=42):
        self.dataset = dataset
        self.root_dir = Path(root_dir)
        self.cells = []
        for i in PPIGraphs(self.root_dir / 'data/processed'):
            self.cells.append(i.cell_name)

        if ppi_depth == 3:
            self.model_path = Path(self.root_dir / Path('models/{}_{}_{}/'.format(
                dataset, split, seed
            )))
            self.ckpt_path = str(list(self.model_path.glob('**/*.ckpt'))[0])
            self.model = MultimodalAttentionNet.load_from_checkpoint(
                self.ckpt_path,
                data_dir=None
            )

        if ppi_depth != 3:
            self.model_path = Path(self.root_dir / Path('models/{}_{}_{}_ppi_depth_{}/'.format(
                dataset, split, seed, ppi_depth
            )))
            self.ckpt_path = str(list(self.model_path.glob('**/*.ckpt'))[0])
            self.model = MultimodalAttentionNet(conf, ppi_depth=ppi_depth, data_dir=None).load_from_checkpoint(
                self.ckpt_path, data_dir=None, ppi_depth=ppi_depth
            )

        self.data_path = Path(root_dir / Path('data/processed/{}_{}/'.format(dataset, split)))
        self.model.eval()

    def top_k_interactions(self, k_interactions=25):
        # interactions in NCI cells
        nci_cells = list(self.labels.loc[self.labels['dataset'] == 'NCI60']['cellosaurus_accession'].unique())
        gdsc_data = self.labels.loc[
            (self.labels['dataset'] == 'GDSC') & (self.labels['dataset'] == 'CTRP')
            ]
        nci_cells = list(gdsc_data.loc[gdsc_data['cellosaurus_accession'].isin(nci_cells)].unique())

        for cell in nci_cells:
            try:
                _, links = self.ppi_graph(cell)
                links.sort_values(by='attention', descending=True, inplace=True)
                top_k = links.iloc[:k_interactions]

            except:
                print('Cell {} not in DepMap'.format(cell))

    def interactions_dict(self, top_k_interactions):
        # use top_k for generating top k links
        def top_k(attention_dict, cell_name, top_k_interactions, mean_attention):
            attention = attention_dict[cell_name]
            attention['difference'] = abs(attention['attention'] - mean_attention)
            attention = attention.sort_values(by='difference', ascending=False).iloc[:top_k_interactions]
            attention['path'] = attention['protein_1'] + '-' + attention['protein_2']
            return list(attention['path'])

        attention, cellosaurus = self.attention_links()  # generate attention for cell lines

        attentions = []
        for i in attention:
            attentions.append(i['attention'].values)

        mean_attention = np.array(attentions).mean(axis=0)
        attention_dict = dict(zip(cellosaurus, attention))
        interactions_list = []
        cellosaurus_accession = []

        for cell in self.cells:
            cell_interactions = top_k(attention_dict, cell, top_k_interactions, mean_attention)
            interactions_list.append(cell_interactions)
            cellosaurus_accession.append(cell)

        interactions_dict = dict(zip(cellosaurus_accession, interactions_list))

        return interactions_dict

    def predict(self, drug_id, smiles, cellosaurus_accession):
        query = InferenceDataset(drug_id=drug_id,
                                 smiles=smiles,
                                 cellosaurus_accession=cellosaurus_accession)

        test_loader = DataLoader(query, 1, shuffle=False, num_workers=8, pin_memory=True,
                                 collate_fn=collate)

        links = pd.read_csv(self.root_dir / 'data/processed/ppi_links.csv', index_col=0)

        for batch in test_loader:
            adj_mat, dist_mat, x = batch[0]
            x_ppi = batch[1].x
            ppi_edge_index = batch[1].edge_index
            ppi_batch = batch[1].batch
            mask = torch.sum(torch.abs(x), dim=-1) != 0
            y_hat, attention = self.model(
                x,
                adj_mat,
                dist_mat,
                mask,
                x_ppi,
                ppi_edge_index,
                ppi_batch,
            )
            y_hat = y_hat.squeeze(-1).detach().cpu().numpy()
            attention = torch.mean(attention[1], dim=1).detach().cpu().numpy()
            links['attention'] = attention

        return y_hat, links
    def attention_links(self):
        cells = self.cells
        attention = []
        cellosaurus_accession = []
        for cell in tqdm(cells):
            #try:
            _, links = self.predict(drug_id=['benzene'],
                                    smiles=["C1=CC=CC=C1"],
                                    cellosaurus_accession=[cell])
            attention.append(links)
            cellosaurus_accession.append(cell)
            #except:
            #print('Cell not in DepMap')

        return attention, cellosaurus_accession

    def eval_test_sets(self, split='random'):
        ap_blind_cells = []
        ap_blind_cells_drugs = []
        ap_blind_drugs = []
        auc_blind_cells = []
        auc_blind_cells_drugs = []
        auc_blind_drugs = []

        test_sets = self.data_path.glob('**/*') #list all file is data_path directory
        blind_cells = list(test_sets)
        print(blind_cells)

        print(error)
        # calculate blind drugs
        for i in blind_drugs:
            ap, auc = self.calculate_metrics(i, split)
            print("AP={} for {} blind drugs on {} split ".format(ap, i, split))
            print("AUC={} for {} blind drugs on {} split".format(auc, i, split))
            ap_blind_drugs.append(ap)
            auc_blind_drugs.append(auc)

        # calculate blind cells
        for i in blind_cells:
            ap, auc = self.calculate_metrics(i, split)
            print("AP={} for {} blind cell on {} split ".format(ap, i, split))
            print("AUC={} for {} blind cell on {} split ".format(auc, i, split))
            ap_blind_cells.append(ap)
            auc_blind_cells.append(auc)

        for i in blind_cells_drugs:
            ap, auc = (self.calculate_metrics(i, split))
            print("AP={} for {} blind cell, blind drugs on {} split ".format(ap, i, split))
            print("AUC={} for {} blind cell, blind drugs on {} split ".format(auc, i, split))
            ap_blind_cells_drugs.append(ap)
            auc_blind_cells_drugs.append(auc)

        if self.dataset == 'NCI60':
            train = pd.read_csv(self.data_path + 'nci_train.csv', index_col=0)
            val = pd.read_csv(self.data_path + 'nci_valid.csv', index_col=0)
            cvcl_0031 = pd.read_csv(self.data_path + '/CVCL_0031_blind_drugs.csv', index_col=0)
            blind_drugs = cvcl_0031['pubchem_cid'].unique()
            cell_lines_train = train['cellosaurus_accession'].unique()
            for cell in cell_lines_train:
                cell_data = self.labels.loc[(self.labels['cellosaurus_accession'] == cell) &
                                            (self.labels['pubchem_cid'].isin(blind_drugs))]
                if split == 'scaffold':
                    cell_data = cell_data.loc[~cell_data['scaffolds'].isin(train['scaffolds'])]
                    cell_data = cell_data.loc[~cell_data['scaffolds'].isin(val['scaffolds'])]
                ap, auc = self.calculate_metrics(cell_data, split)
                print("AP={} for {} blind drugs on {} split ".format(ap, cell, split))
                print("AUC={} for {} blind drugs on {} split ".format(auc, cell, split))
                ap_blind_drugs.append(ap)
                auc_blind_drugs.append(auc)

        print('\n')
        print(
            "Blind cells test with {} split: AUC={}; AP={}".format(
                split, np.mean(auc_blind_cells), np.mean(ap_blind_cells)
            )
        )

        print(
            "Blind drugs test with {} split: AUC={}; AP={}".format(
                split, np.mean(auc_blind_drugs), np.mean(ap_blind_drugs)
            )
        )

        print(
            "Blind cells, blind drugs, test with {} split: AUC={}; AP={}".format(
                split, np.mean(auc_blind_cells_drugs), np.mean(ap_blind_cells_drugs)
            )
        )

    def calculate_metrics(self, data, split='random'):
        try:
            dataset = PairDataset(self.data_path + data)
            test_data = pd.read_csv(self.data_path + data)
        except:
            dataset = PairDataset(data)
            test_data = data

        test = DataLoader(dataset, 32, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate)
        predictions = []
        drug_name = []
        cell_name = []

        for batch in test:
            x = batch[0].x
            edge_index = batch[0].edge_index
            mol_batch = batch[0].batch
            edge_attr = batch[0].edge_attr
            x_ppi = batch[1].x
            ppi_edge_index = batch[1].edge_index
            ppi_batch = batch[1].batch
            y_hat, attention, attention_ppi = self.model(
                x,
                edge_index,
                edge_attr,
                mol_batch,
                x_ppi,
                ppi_edge_index,
                ppi_batch,
            )
            y_hat = y_hat.squeeze(-1)
            predictions.append(y_hat.cpu().detach().numpy().flatten().tolist())
            drug_name.append(batch[0].drug_name)
            cell_name.append(batch[1].cell_name)
        predictions = [item for sublist in predictions for item in sublist]
        drug_name = [item for sublist in drug_name for item in sublist]
        cell_name = [item for sublist in cell_name for item in sublist]

        prediction_df = pd.DataFrame({'predictions': predictions,
                                      'pubchem_cid': drug_name,
                                      'cellosaurus_accession': cell_name})
        train_data = pd.read_csv(self.data_path + 'nci_train.csv')
        val_data = pd.read_csv(self.data_path + 'nci_valid.csv')
        prediction_df = prediction_df.merge(test_data[['cellosaurus_accession',
                                                       'pubchem_cid',
                                                       'sensitivity_uM']], how='inner',
                                            on=['pubchem_cid', 'cellosaurus_accession'])
        if split == 'scaffold':
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(train_data['scaffolds'])]
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(val_data['scaffolds'])]
        overlap = prediction_df.merge(train_data, how='inner', on=['pubchem_cid', 'cellosaurus_accession'])
        prediction_df = prediction_df.loc[~prediction_df['pubchem_cid'].isin(overlap['pubchem_cid'])]

        ap = average_precision_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
        roc_auc = roc_auc_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
        return ap, roc_auc
