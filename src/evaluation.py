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
        id_blind_cells = []
        ap_blind_cells = []
        auc_blind_cells = []

        id_double_blind = []
        ap_double_blind = []
        auc_double_blind = []

        id_blind_drugs = []
        ap_blind_drugs = []
        auc_blind_drugs = []

        test_sets = list(self.data_path.glob('**/*')) #list all file is data_path directory
        blind_cells = [test for test in test_sets if str(test).endswith('blind_cells.csv')][0]
        blind_drugs = [test for test in test_sets if str(test).endswith('blind_drugs.csv')][0]
        double_blind = [test for test in test_sets if str(test).endswith('double_blind.csv')][0]

        # calculate blind drugs
        blind_cells = pd.read_csv(blind_cells, index_col=0)
        cell_ids = blind_cells['cellosaurus_accession'].unique()
        for i in cell_ids:
            ap, auc = self.calculate_metrics(i, split)
            id_blind_cells.append(i)
            ap_blind_cells.append(ap)
            auc_blind_cells.append(auc)

        # calculate blind cells
        blind_drugs = pd.read_csv(blind_drugs, index_col=0)
        for i in blind_drugs:
            ap, auc = self.calculate_metrics(i, split)
            id_blind_drugs.append(i)
            ap_blind_drugs.append(ap)
            auc_blind_drugs.append(auc)

        double_blind = pd.read_csv(double_blind, index_col=0)
        for i in double_blind:
            ap, auc = (self.calculate_metrics(i, split))
            id_double_blind.append(i)
            ap_double_blind.append(ap)
            auc_double_blind.append(auc)

    def calculate_metrics(self, data, split='random'):
        """Calculate metrics on CPU if there are many elements to prevent memory errors on the GPu"""
        """
        try:
            dataset = PairDataset(self.data_path + data)
            test_data = pd.read_csv(self.data_path + data)
        except:
            dataset = PairDataset(data)
            test_data = data
        """

        dataset = PairDataset(self.data_path + data)
        test_data = pd.read_csv(self.data_path + data)

        test = DataLoader(dataset, 32, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate)
        predictions = []
        drug_name = []
        cell_name = []

        for batch in test:
            adj_mat, dist_mat, x = batch[0]
            x_ppi = batch[1].x
            ppi_edge_index = batch[1].edge_index
            ppi_batch = batch[1].batch
            mask = torch.sum(torch.abs(x), dim=-1) != 0
            y_hat, _ = self.model(
                x,
                adj_mat,
                dist_mat,
                mask,
                x_ppi,
                ppi_edge_index,
                ppi_batch,
            )
            y_hat = y_hat.squeeze(-1).detach().cpu().numpy()
            predictions.append(y_hat.cpu().detach().numpy().flatten().tolist())

        predictions = [item for sublist in predictions for item in sublist]
        prediction_df = pd.DataFrame({'predictions': predictions,
                                      'pubchem_cid': test_data['pubchem_cid'],
                                      'cellosaurus_accession': test_data['cellosaurus_accession'],
                                      'sensitivity_uM': test_data['sensitivity_uM'],
                                      'scaffolds': test_data['scaffolds']})
        train_data = pd.read_csv(self.data_path / 'train.csv', index_col=0)
        val_data = pd.read_csv(self.data_path / 'valid.csv', index_col=0)
        if split == 'scaffold':
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(train_data['scaffolds'])]
            prediction_df = prediction_df.loc[~prediction_df['scaffolds'].isin(val_data['scaffolds'])]

        ap = average_precision_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
        roc_auc = roc_auc_score(prediction_df['sensitivity_uM'], prediction_df['predictions'])
        return ap, roc_auc

    def overlaps(self):
        overlap = []
        cells = []
        for key in dict1:
            list1 = dict1[key]
            list2 = dict2[key]
            overlap.append(len([value for value in list1 if value in list2]))
            cells.append(key)
        return cells, overlap

    def disease_dicts(self):
        def disease_overlap(sample_info, dict1):
            new_dict = dict1
            sample_info = sample_info.loc[sample_info['RRID'].isin(list(new_dict.keys()))]
            primary_disease_list = sample_info['Subtype'].unique()
            disease_list = []
            interactions_list = []
            overlap = []
            for disease in primary_disease_list:
                disease_specific_ccl = sample_info.loc[sample_info['Subtype'] == disease]['RRID'].unique()
                num_cells = len(disease_specific_ccl)
                print(disease)
                print(num_cells)
                disease_interactions = []
                for ccl in disease_specific_ccl:
                    new_dict[ccl] = new_dict[ccl][0:10]
                    disease_interactions.append(new_dict[ccl])

                disease_interactions = [item for sublist in disease_interactions for item in sublist]
                n_interactions = [0] * len(disease_interactions)
                interactions_dict = dict(zip(disease_interactions, n_interactions))

                # calculate num interactions
                for ccl in disease_specific_ccl:
                    interactions = new_dict[ccl]
                    for i in interactions:
                        if i in interactions_dict.keys():
                            interactions_dict[i] += 1

                """
                #calculate % interactions
                for ccl in disease_specific_ccl:
                    interactions = new_dict[ccl]
                    for i in interactions:
                        if i in interactions_dict.keys():
                            n = interactions_dict[i]
                            interactions_dict[i] = (n / num_cells) * 100 # calculate percentage
                """

                interactions_list.append(interactions_dict)
                disease_list.append(disease)

            return dict(zip(disease_list, interactions_list))

