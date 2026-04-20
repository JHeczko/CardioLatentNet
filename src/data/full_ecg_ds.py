import torch
import pandas as pd
import ast
import wfdb
from typing import Literal
from sklearn.preprocessing import MultiLabelBinarizer

class Full_ECG_DataSet(torch.utils.data.Dataset):

    def __init__(self, path:str,
                 sampling_rate:Literal[100,500]=100,
                 mode:Literal['train','val','test']='train',
                 target: Literal["sex", "age", "scp"]="scp"):

        super().__init__()

        self.__test_fold = 10
        self.__val_fold = 9

        assert mode in ['train','val','test']
        assert sampling_rate in [100,500]
        assert type(target) is str
        assert target in ["sex", "age", "scp"]

        self.sampling_rate = sampling_rate
        self.mode = mode
        self.target = target
        self.path = path

        # =-=-=-=-=-=-=-=-= load data =-=-=-=-=-=-=-=-=
        Y: pd.DataFrame = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id', na_values='-1')
        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)

        # =-=-=-=-=-=-=-=-= filtering data from NaN =-=-=-=-=-=-=-=-=
        # ======== AGE ========
        if target == 'age':
            Y = Y[Y['age'] != -1]
        # ======== sex ========
        if target == 'sex':
            Y = Y[Y['sex'] != -1]

        # =-=-=-=-=-=-=-=-= data splitting based on mode =-=-=-=-=-=-=-=-=
        if mode == 'train':
            Y = Y[~Y.strat_fold.isin([self.__test_fold, self.__val_fold])]
        elif mode == 'val':
            Y = Y[Y.strat_fold == self.__val_fold]
        elif mode == 'test':
            Y = Y[Y.strat_fold == self.__test_fold]
        else:
            raise ValueError('Invalid mode (train|val|test)')

        # =-=-=-=-=-=-=-=-= Loading time serieses =-=-=-=-=-=-=-=-=
        # only loading to X the file paths for specific sampling_rate, for RAM efficiency
        if sampling_rate == 100:
            X = Y['filename_lr'].values
            #data = [wfdb.rdsamp(path + f) for f in Y.filename_lr]
        else:
            X = Y['filename_hr'].values
            #data = [wfdb.rdsamp(path + f) for f in Y.filename_hr]
        #X = np.array([signal for signal, meta in data])


        # =-=-=-=-=-=-=-=-= load targets =-=-=-=-=-=-=-=-=
        if target == "scp":
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            agg_df = agg_df[agg_df.diagnostic == 1]

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
        elif target == "sex":
            y = Y.sex.to_numpy().reshape(-1, 1)
        elif target == "age":
            y = Y.age.to_numpy().reshape(-1, 1)

        self.X_files = X
        self.y = y

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        signal, _ = wfdb.rdsamp(self.path + self.X_files[idx])

        return torch.tensor(signal.copy(), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
