import os
import pandas as pd

def make_datalist(cfg):
    csv_path = os.path.join(cfg.DATA.split_fold, 'splits_{}.csv'.format(cfg.DATA.fold_idx))
    print(csv_path)
    all_splits = pd.read_csv(csv_path)
    train_list = all_splits.iloc[:, 0:1] #get_split_from_df(all_splits, 'train')
    train_list = list((train_list.values).squeeze())
    valid_list = all_splits.iloc[:, 1:2]#get_split_from_df(all_splits, 'val')
    valid_list = list((valid_list.values).squeeze())
    test_list = all_splits.iloc[:, 2:3]#get_split_from_df(all_splits, 'test')
    test_list = list((test_list.values).squeeze())
    for m in range(len(valid_list)):
        if isinstance(valid_list[m], float):
            valid_list = valid_list[:m]
            break
    for m in range(len(test_list)):
        if isinstance(test_list[m], float):
            test_list = test_list[:m]
            break
    for m in range(len(train_list)):
        if isinstance(train_list[m], float):
            train_list = train_list[:m]
            break
    # 防止出现空值
    datalsit = {}
    datalsit['train'] = train_list
    datalsit['eval'] = valid_list
    datalsit['test'] = test_list

    return datalsit