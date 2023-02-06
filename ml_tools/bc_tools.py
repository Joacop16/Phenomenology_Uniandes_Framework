import numpy as np
import pandas as pd
import os

def _get_features(self,channels):
    sets=[set(self._get_features_by_key(channel)) for channel in channels]
    common_features=sets[0]
    for set_ in sets:
        common_features = common_features.intersection(set_)
    self.features=list(common_features)

def prepare_features(data_dict,name='',channel="",features=[]):
    signal_feature=[]
    for i in range(data_dict[channel][name].shape[0]):
        row=[data_dict[channel][name][feature][i] for feature in features ]        
        signal_feature.append(row)
    return np.array(signal_feature)

def concat_signals(data_dict,signal_list,channel,features):
    names=signal_list.copy()
    name_0=names.pop(0)
    arr=prepare_features(data_dict,name_0,channel,features)[:,:]
    for name in names:
        arr=np.concatenate(
            (arr , prepare_features(data_dict ,name ,channel , features)[:,:] )
        )
    np.random.shuffle(arr)
    return arr

def concat_channels(folder,signal_list,channels,features):
    def read_csv(ch):
        dict_={}
        for signal in signal_list:
            try:
                df=pd.read_csv(
                    os.path.join(folder,f"{signal}_{ch}.csv")
                )
                dict_.update(
                    {signal : df}
                )
            except:
                continue
        
        return ch, dict_
    
    data_dict=dict(map(read_csv,channels))
    chs=channels.copy()
    ch_0=chs.pop(0)
    arr=concat_signals(data_dict,signal_list,ch_0,features)
    for channel in chs:
        arr=np.concatenate(
            (arr,concat_signals(data_dict,signal_list,channel,features)[:,:])
        )
    np.random.shuffle(arr)
    return arr

def prepare_to_train (signal,bkg):
    bkg_vsize=min([len(signal),len(bkg)])
    signal_vsize=bkg_vsize

    pred=signal[:signal_vsize, :]
    pred=np.concatenate((pred,bkg[:bkg_vsize, :]))
    
    labels=np.zeros(np.shape(pred)[0])
    labels[:signal_vsize] = 1
    
    return pred , labels

def get_yield(csv_files_path,signal_list,channel):
    df=pd.read_csv(
        os.path.join(
            csv_files_path,
            f"Cutflow_{channel}.csv"
        )
    )
    yields=0
    for name in signal_list:
        try:
            yields+=df[name][(df.shape[0]-1)]
        except:
            continue
    return yields

def get_yields(csv_files_path,names,channels):
    dict_={}
    for channel in channels:
        dict_[channel]={}
        for name in names: 
            dict_[channel][name]=get_yield(csv_files_path,[name],channel)
    return dict_