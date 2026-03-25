import torch
import pandas as pd
import ast
import wfdb
import numpy as np
from typing import Literal
from sklearn.preprocessing import MultiLabelBinarizer

class Full_ECG_DataSet(torch.utils.data.Dataset):

    def __init__(self, path:str,
                 sampling_rate:Literal[100,400]=100,
                 mode:Literal['train','val','test']='train',
                 target: Literal["sex", "age", "scp"]="scp"):

        super().__init__()

        self.__test_fold = 10
        self.__val_fold = 9

        assert mode in ['train','val','test']
        assert sampling_rate in [100,400]
        assert type(target) is str
        assert target in ["sex", "age", "scp"]

        self.sampling_rate = sampling_rate
        self.mode = mode
        self.target = target
        self.path = path

        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)

        if mode == 'train':
            Y = Y[~Y.strat_fold.isin([self.__test_fold, self.__val_fold])]
        elif mode == 'val':
            Y = Y[Y.strat_fold == self.__val_fold]
        elif mode == 'test':
            Y = Y[Y.strat_fold == self.__test_fold]
        else:
            raise ValueError('Invalid mode (train|val|test)')

        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # # Load raw signal data
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in Y.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in Y.filename_hr]
        X = np.array([signal for signal, meta in data])

        # Load scp_statements.csv for diagnostic aggregati

        name_to_index = {v: i for i, v in enumerate(agg_df.diagnostic_class.unique())}
        labels = []
        for codes in Y.scp_codes:
            tmp = []
            for key, value in codes.items():
                if (key in agg_df.index) and (value > 0.5):
                    tmp.append(name_to_index[agg_df.loc[key].diagnostic_class])
            labels.append(tmp)

        # transform labels to ONEHOT (doing classification)
        y = MultiLabelBinarizer().fit_transform(labels)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.bfloat16), torch.tensor(self.y[idx], dtype=torch.bfloat16)

