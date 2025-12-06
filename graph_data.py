import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from utils import load_GO_annot


def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    # feats = [item[0] for item in graphs]
    # lengths = [item[1] for item in batch]
    # labels = [item[2] for item in batch]
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()


class GoTermDataset(Dataset):

    def __init__(self, set_type, task):
        # task can be among ['bp','mf','cc']
        self.task = task
        prot2annot, goterms, gonames, counts = load_GO_annot('/home/415/hz_project/HEAL/data/nrPDB-GO_2019.06.18_annot.tsv')

        goterms = goterms[self.task]
        gonames = gonames[self.task]
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))

        self.pos_weights = torch.tensor(pos_weights).float()

        self.processed_dir = '/home/415/hz_project/data_processed/'

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_processed_graph_pt5.pt"))

        self.pdbch_list = torch.load(os.path.join('/home/415/hz_project/HEAL/data/processed/', f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]

        self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        self.y_true = torch.tensor(self.y_true)

        ss_feat_npz = np.load(os.path.join('/home/415/hz_project/HEAL/data/processed/', f"{set_type}_ss.npz"))

        # 给 PDB 模型图添加 ss_feat
        for i, graph in enumerate(self.graph_list[:len(self.pdbch_list)]):
            pdbid = self.pdbch_list[i]
            if pdbid in ss_feat_npz:
                ss_feat = ss_feat_npz[pdbid]
                if ss_feat.shape[0] == graph.x.shape[0]:
                    graph.ss_feat = torch.tensor(ss_feat, dtype=torch.float32)
                else:
                    print(f"[PDB] Mismatch for {pdbid}: {ss_feat.shape[0]} != {graph.x.shape[0]}")
            else:
                print(f"[PDB] Missing ss_feat for {pdbid}")

        # AF
        prot2annot1, goterms1, gonames1, counts1 = load_GO_annot("/home/415/hz_project/HEAL/data/nrSwiss-Model-GO_annot.tsv")

        graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_processed_graph_pt5.pt"))
        self.pdbch_list_af = torch.load(os.path.join('/home/415/hz_project/HEAL/data/processed/', f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]

        ss_feat_npz_af = np.load(os.path.join('/home/415/hz_project/HEAL/data/processed/', f"AF2{set_type}_ss.npz"))
        for i, graph in enumerate(graph_list_af):
            pdbid = self.pdbch_list_af[i]
            if pdbid in ss_feat_npz_af:
                ss_feat = ss_feat_npz_af[pdbid]
                if ss_feat.shape[0] == graph.x.shape[0]:
                    graph.ss_feat = torch.tensor(ss_feat, dtype=torch.float32)
                else:
                    print(f"[AF2] Mismatch for {pdbid}: {ss_feat.shape[0]} != {graph.x.shape[0]}")
            else:
                print(f"[AF2] Missing ss_feat for {pdbid}")
        
        self.graph_list += graph_list_af
        y_true_af = np.stack([prot2annot1[pdb_c][self.task] for pdb_c in self.pdbch_list_af])

        self.y_true = np.concatenate([self.y_true, y_true_af],0)
        self.y_true = torch.tensor(self.y_true)


    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)
