from concurrent.futures import ProcessPoolExecutor

import torch
import pandas as pd
import ast
import wfdb
from typing import Literal
from sklearn.preprocessing import MultiLabelBinarizer
import neurokit2 as nk
from torch.utils.data import DataLoader

from tqdm import tqdm

import timeit

def process_file(args):
    file, label, path, sampling_rate, pre_sample, post_sample = args

    out = []
    series, _ = wfdb.rdsamp(path + file)
    II_series = series[:, 1]

    cleaned_series = nk.ecg_clean(II_series, sampling_rate=sampling_rate)
    _, info = nk.ecg_peaks(cleaned_series, sampling_rate=sampling_rate)
    r_peaks = info["ECG_R_Peaks"]

    for peak in r_peaks:
        if (peak - pre_sample >= 0) and (peak + post_sample <= len(II_series)):
            out.append((file, peak, label))

    return out

class Hearbeat_ECG_DataSet(torch.utils.data.Dataset):

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
        Y: pd.DataFrame = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)

        # =-=-=-=-=-=-=-=-= filtering data from NaN =-=-=-=-=-=-=-=-=
        # ======== AGE and SEX ========
        if target in ['age', 'sex']:
            Y = Y.dropna(subset=[target])

        # =-=-=-=-=-=-=-=-= data splitting based on mode =-=-=-=-=-=-=-=-=
        if mode == 'train':
            Y = Y[~Y.strat_fold.isin([self.__test_fold, self.__val_fold])]
        elif mode == 'val':
            Y = Y[Y.strat_fold == self.__val_fold]
        elif mode == 'test':
            Y = Y[Y.strat_fold == self.__test_fold]
        else:
            raise ValueError('Invalid mode (train|val|test)')


        # =-=-=-=-=-=-=-=-= load targets =-=-=-=-=-=-=-=-=
        if target == "scp":
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            agg_df = agg_df[agg_df.diagnostic == 1]

            name_to_index = {v: i for i, v in enumerate(agg_df.diagnostic_class.unique())}
            labels = []
            for codes in tqdm(Y.scp_codes, desc="Processing SCP codes"):
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


        #  =-=-=-=-=-=-=-=-= Calculating length and peaks =-=-=-=-=-=-=-=-=
        if sampling_rate == 100:
            X = Y['filename_lr'].values
        else:
            X = Y['filename_hr'].values

        self.pre_sample = int(0.2 * self.sampling_rate)
        self.post_sample = int(0.4 * self.sampling_rate)

        self.X_files = []

        args_list = [
            (file, label, self.path, self.sampling_rate, self.pre_sample, self.post_sample)
            for file, label in zip(X, y)
        ]

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_file, args_list),
                total=len(args_list),
                desc="Processing ECG files",
                unit="file"
            ))

        for res in results:
            self.X_files.extend(res)


    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        file, peak, label = self.X_files[idx]

        signal_heartbeat, _ = wfdb.rdsamp(self.path + file, sampfrom=peak-self.pre_sample, sampto=peak+self.post_sample)

        return torch.tensor(signal_heartbeat.copy(), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    time_start = timeit.default_timer()

    ds = Hearbeat_ECG_DataSet(path="../../dataset/ptb_xl/")
    loader = DataLoader(ds, batch_size=10, shuffle=True)

    print(f"Dataset len: {len(ds)}")

    time_end = timeit.default_timer()

    print("Time elapsed:", time_end - time_start)