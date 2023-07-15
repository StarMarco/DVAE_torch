import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans 
import pandas as pd 
import numpy as np 
from data.utils import * 

class DataPrep():
    def __init__(self, main_PATH, dataset):
        """
        Inputs:
            dataset (str): can be either FD00x (x=1,2,3 or 4)
                to pick which CMAPSS dataset is loaded.
        """    
        self.dataset = dataset
        # get data 
        train_path = main_PATH + "\\train_" + dataset + ".txt"
        test_path = main_PATH + "\\test_" + dataset + ".txt"
        testRUL_path = main_PATH + "\\RUL_" + dataset +  ".txt"

        train = pd.read_csv(train_path, parse_dates=False, delimiter=" ", decimal=".", header=None)
        test  = pd.read_csv(test_path, parse_dates=False, delimiter=" ", decimal=".", header=None)
        RUL = pd.read_csv(testRUL_path, parse_dates=False, decimal=".", header=None)

        tableNA = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1)
        tableNA.columns = ["train", "test"]

        # drop the columns that consist of missing values 
        train.drop(train.columns[[-1,-2]], axis=1, inplace=True)    
        test.drop(test.columns[[-1,-2]], axis=1, inplace=True)      

        cols = ['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'
                , 's9','s10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

        train.columns = cols
        test.columns = cols

        train = pd.merge(train, train.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
        train.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
        train["TTF"] = train["maxcycles"] - train["cycles"]

        test = pd.merge(test, test.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
        test.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)

        # add to the class variables 
        self.train = train 
        self.test = test 
        self.RUL = RUL 

    def op_normalize(self, K=1): 
        """
        Normalizes the data based on operating mode clusters catogorized and found with 
        K-means clustering. 

        x_n = (x - op_mean) / op_std 

        This is done on the training data and the means and stdevs are saved and used to 
        normalize the testing data to keep consistant. 

        See the following paper for more detials on the theory this is based on,  
        
        Pasa, G., Medeiros, I., Yoneyama, T. (2019). 
        Operating Condition-Invariant Neural Network-based Prognostics Methods applied on Turbofan Aircraft Engines. 
        Annual Conference of the PHM Society, 1–10. 
        https://doi.org/https://doi.org/10.36001/phmconf.2019.v11i1.786

        where: 
            x = sensor value 
            op_mean = mean of the operating mode cluster "x" is in 
            op_std = standard deviation of the operating mode cluster "x" is in 
            x_n = normalized sensor value 

        Inputs:
            K: Type = int 
                amount of clusters/operating modes for K-means clustering 
                *Note if K=1 then this is standard normalization 

        Outputs: 
            data_norm: Type, Pandas DataFrame 
                same as the input dataframe but the sensor values are 
                normalized based on the mean and standard deviation of the operating 
                class they are in 
        """
        data = self.train
        data_test = self.test

        # K-means clustering 
        op_set = [col for col in data.columns if col.startswith("op")] 
        X = data[op_set].values
        self.kmeans = KMeans(n_clusters=K, n_init=10).fit(X)    # cluster using training data 

        # Append operating cluster id's to dataset 
        data_op = self.operating_modes(data)
        data_op_test = self.operating_modes(data_test)

        # operating modes needed to loop over 
        self.clusters = data_op["op_c"].max()

        # copy for later normalization
        data_n = data_op.copy()
        data_n_test = data_op_test.copy()

        # find the means and standard deviations of sensors in each opperating mode of training data 
        sensors = [col for col in data_n.columns if col.startswith("s")] 
        self.means = []
        self.stds = [] 
        for c in range(0, self.clusters+1):
            sens = data_n[(data_n.op_c==c)][sensors]
            mean = sens.mean()
            std = sens.std()

            self.means.append(mean)
            self.stds.append(std)

        # use these means and stadard deviations to normalize the sensor values 
        drop_index = [[0]]
        for s in self.stds:
            if len(drop_index[0]) < len(s[s<1e-8]): # find the largest amount of sensors with a small stdev 
                drop_index = s[s<1e-8].index 

        if len(drop_index) == 1:
            if (drop_index) == [[0]]:
                None 
        else: 
            # drop sensor values with small standard deviation 
            for i in range(len(self.means)):
                self.means[i].drop(drop_index, inplace=True)
                self.stds[i].drop(drop_index, inplace=True)

            data_n = self.drop_sensors(data_n, drop_index)
            data_n_test = self.drop_sensors(data_n_test, drop_index)

            # drop sensors with 2 or less unique values 
            drop_index = self.drop_same(data_n)
            for i in range(len(self.means)):
                self.means[i].drop(drop_index, inplace=True)
                self.stds[i].drop(drop_index, inplace=True)

            data_n = self.drop_sensors(data_n, drop_index)
            data_n_test = self.drop_sensors(data_n_test, drop_index)

        # normalize the sensors for each unit in the dataset 
        self.ntrain = self.norm(data_n)
        self.ntest = self.norm(data_n_test)

    def operating_modes(self, data):
        """
        Use K-means to classify data into operating modes 
        """
        op_set = [col for col in data.columns if col.startswith("op")]
        X = data[op_set].values
        kmeans_pred = self.kmeans.predict(X)

        # append operating mode classifications to data 
        op_cluster = pd.DataFrame({"op_c": kmeans_pred})
        data_op = pd.concat([data, op_cluster], axis=1)

        return data_op

    def drop_sensors(self, data, drop_index):
        """
        Drops sensors based on index given (drop_index)
        """
        data.drop(drop_index, axis=1, inplace=True)

        return data

    def drop_same(self, data):
        """
        Returns the index to drop the sensors with only 2 or less unique values 
        in the series 
        """
        sensors = [col for col in data.columns if col.startswith("s")]
        drop_index = data[sensors].loc[:,data[sensors].nunique() < 3].columns

        return drop_index

    def norm(self, data):
        """
        normalize data based on the mean and standard deviation of the 
        operating condition it is apart of 
        """
        units = int(data["unit"].max())
        sensors = [col for col in data.columns if col.startswith("s")]
        for unit in range(1, units+1):
            for c in range(0, self.clusters+1):
                sens = data[(data.op_c==c) & (data.unit==unit)][sensors]
                sens = (sens - self.means[c]) / self.stds[c]
                data.loc[(data.op_c==c) & (data.unit==unit), sensors] = sens

        return data 

    def prep_data(self, df, T, stride):
        """
        Inputs:
            df (Dataframe): dataframe with training inputs 
            T (int): Time window size of the signals cut from the input sensor signals 
            stride (int): the number of points moved as the time window slides 

        Outputs: 
            x (tensor): input sequences, size (*, seq, x_dim)
            y (tensor): corresponding targets (RULs), size (*, seq, 1)
            t (tensor): corresponding time points, size (*, seq, 1)
        """
        sensors = [col for col in df.columns if col.startswith("s")]

        x = torch.tensor([])
        y = torch.tensor([])
        t = torch.tensor([])
        for unit in range(1, max(df.unit)+1):
            rul = df[df.unit==unit].TTF.values
            sen = df[df.unit==unit][sensors].values
            time = df[df.unit==unit].cycles.values

            # turn the current unit signal into a tensor of size (seq, dim)
            sen = torch.tensor(sen)
            rul = torch.tensor(rul).unsqueeze(-1)
            time = torch.tensor(time).unsqueeze(-1)

            # split entire sequences into 'N' time series blocks of size T, (N, T, dim)
            sen = sen.unfold(0, T, stride).permute(0,2,1)
            rul = rul.unfold(0, T, stride).permute(0,2,1)
            time = time.unfold(0, T, stride).permute(0,2,1)

            # if rul > 130 let rul = 130 if it is below 130 then rul = rul 
            rul = (rul > 130) * 130 + (rul <= 130) * rul

            x = torch.cat((x, sen), dim=0)
            y = torch.cat((y, rul), dim=0)
            t = torch.cat((t, time), dim=0)

        return x, y, t

    def valid_set(self, x_train, y_train, t_train, split=0.2):
        """
        Inputs: 
            x_train (tensor): training input tensor, size (*, seq, x_dim)
            y_train (tensor): training target tensor, size (*, seq, 1)
            t_train (tensor): corresponding training time points, size (*, seq, 1)
            split (float): a number between 0 and 1 to determine the % of data to be converted into a validation set 

        Outputs:
            x_train (tensor): new split training input tensor
            y_train (tensor): new split training targets 
            t_train (tensor): new split corresponding training time points 
            x_valid (tensor): validation inputs split from the original training dataset 
            y_valid (tensor): validation targets split from the original training dataset 
            t_valid (tensor): validation time points split from the original training dataset 
        """
        rng = np.random.default_rng()

        valid_size = int(split * x_train.shape[0])
        total_No_idxs = x_train.shape[0] - 1
        # get a list of random integers to serve as indicies to extract a validation dataset 
        valid_set = list(rng.choice(total_No_idxs, size=valid_size, replace=False))

        # get the remaining possible integers to serve as the indicies for the training dataset 
        train_set = list(np.linspace(0, x_train.shape[0]-1, x_train.shape[0]))
        train_set = [x for x in train_set if x not in valid_set]

        # store new training and validation tensors/datasets
        x_valid = x_train[valid_set]
        y_valid = y_train[valid_set]
        t_valid = t_train[valid_set]
        x_train = x_train[train_set]
        y_train = y_train[train_set]
        t_train = t_train[train_set]

        return x_train, y_train, t_train, x_valid, y_valid, t_valid

    def prep_test(self, df, RUL):
        """
        Prepares input test data used once the network is trained 

        Inputs:
            df (pandas DataFrame): the dataframe with the input sensor data for each unit 
            RUL (pandas DataFrame): the dataframe with the corresponding RUL for each unit 

        Outputs:
            x (list of tensors): list of testing unit sensor tensors, tensor size (1, seq, inputs)
            y (list of tensors): list of testing unit RUL tensors at each time point, tensor size (seq)
            t (list of tensors): list of corresponding time tensors, tensor size (units, seq, 1)
        """
        sensors = [col for col in df.columns if col.startswith("s")]
        RUL = self.RUL.values

        x = [] 
        y = [] 
        t = [] 
        for unit in range(1, max(df.unit)+1):
            sen = df[df.unit==unit][sensors].values
            rul_T = RUL[unit-1]
            time = df[df.unit==unit].cycles.values

            seq = sen.shape[0]          # observed sequence length 
            total_len = rul_T + seq     # total length if run to end of life

            rul = np.linspace(total_len-1, rul_T, seq)

            sen = torch.tensor(sen)
            rul = torch.tensor(rul)
            time = torch.tensor(time)

            rul = (rul > 130) * 130 + (rul <= 130) * rul

            x.append(sen.unsqueeze(0))
            y.append(rul.unsqueeze(0))
            t.append(time.unsqueeze(-1).unsqueeze(0))

        return x, y, t    

    def get_dataloaders(self, bs, x_train, y_train, x_valid, y_valid):
        """
        Inputs:
            bs (int): batch size of the outputs 
            x_train (tensor): training input tensor, size (*, seq, x_dim)
            y_train (tensor): training target tensor, size (*, seq, 1)
            x_valid (tensor): validation inputs split from the original training dataset, size (*, seq, x_dim)
            y_valid (tensor): validation targets split from the original training dataset, size (*, seq, 1)

        Outputs: 
            train_loader (dataloader): dataloader containing the training inputs and targets, size (bs, seq, x_dim) and (bs, seq, 1)
            valid_loader (dataloader): dataloader containing the validation inputs and targets, size (bs, seq, x_dim) and (bs, seq, 1)
        """
        train_DataSet = torch.utils.data.TensorDataset(x_train, y_train) 
        train_loader = torch.utils.data.DataLoader(train_DataSet, shuffle=True, batch_size=bs) 

        valid_DataSet = torch.utils.data.TensorDataset(x_valid, y_valid) 
        valid_loader = torch.utils.data.DataLoader(valid_DataSet, batch_size=bs) 

        return train_loader, valid_loader 

    def get_training_units_data(self, df):
        sensors = [col for col in df.columns if col.startswith("s")]

        xs = []
        ys = []
        ts = [] 
        ops = [] 
        for unit in range(1, max(df.unit)+1):
            rul = np.expand_dims(df[df.unit==unit].TTF.values, axis=-1)
            sen = df[df.unit==unit][sensors].values
            time = np.expand_dims(df[df.unit==unit].cycles.values, axis=-1)
            op = df[df.unit==unit].op_c.values
            
            # if rul > 130 let rul = 130 if it is below 130 then rul = rul 
            rul = (rul > 130) * 130 + (rul <= 130) * rul

            xs.append(torch.tensor(sen))
            ys.append(torch.tensor(rul))
            ts.append(torch.tensor(time))
            ops.append(torch.tensor(op))

        return xs, ys, ts, ops

    def semi_supervised_dataprep(self, T, bs, percent_unlabelled, valid_split, stride=1):
        dataset = self.dataset
        if dataset == "FD001" or dataset == "FD003":
            self.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
        else: 
            self.op_normalize(K=6) 

        x_train, y_train, t_train = self.prep_data(self.ntrain, T, stride)
        x_test, y_test, t_test = self.prep_test(self.ntest, self.RUL)

        split = percent_unlabelled / 100.

        rng = np.random.default_rng()
        ulb_idxs = rng.choice(x_train.shape[0], size=int(x_train.shape[0] * split), replace=False)

        semi_ds = SemiDataset(x_train, y_train, ulb_idxs)
        train_loader = DataLoader(semi_ds, bs)

        # recreate new dataset with only a certain % of labels 
        xs = torch.tensor([])
        ys = torch.tensor([])
        for x, y in train_loader:
            xs = torch.cat((x, xs), dim=0)
            ys = torch.cat((y, ys), dim=0)

        un_xs, un_ys, _, x_valid, y_valid, _ = self.valid_set(xs, ys, torch.randn(xs.shape), split=valid_split)

        valid_ds = TensorDataset(x_valid, y_valid)
        valid_loader = DataLoader(valid_ds, bs)

        train_ds = TensorDataset(un_xs, un_ys)
        train_loader = DataLoader(train_ds, bs, shuffle=True)   

        return train_loader, valid_loader, x_test, y_test, t_test 
                
class SemiDataset(Dataset):
    def __init__(self, x_data, y_data, ulb_idxs):
        self.x_data = x_data 
        self.y_data = y_data 
        self.ulb_idxs = ulb_idxs 

    def __len__(self):
        return self.x_data.size(0)
    
    def __getitem__(self, index):
        x = self.x_data[index]
        if index in self.ulb_idxs:
            y = -1 * torch.ones(self.y_data[index].shape)   # replace with -1's
        else: 
            y = self.y_data[index]

        return x, y 
