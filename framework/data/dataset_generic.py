from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import glob
import h5py
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler, Sampler
from data.data_utils import generate_split, nth


# ==========================================
# 辅助工具类与函数
# ==========================================

class SubsetSequentialSampler(Sampler):
    """按顺序采样指定索引的元素"""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    """计算平衡采样权重"""
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)


def collate_MIL(batch):
    """
    自定义 batch 整理函数
    返回: [features, slide_labels, coords, slide_ids, case_labels]
    """
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    slide_ids = [item[3] for item in batch]
    case_labels = torch.LongTensor([item[4] for item in batch])

    # 处理坐标 (兼容 list 或 tensor)
    coords = torch.cat([torch.tensor(item[2], dtype=torch.float32).unsqueeze(0) for item in batch], dim=0)

    return [img, label, coords, slide_ids, case_labels]


def save_splits_stats(split_datasets, label_map=None):
    """统计切分后的类别分布"""
    train_split = split_datasets[0].slide_data
    # 简单的类别计数
    counts = train_split['slide_label'].value_counts().sort_index()
    return counts.tolist()


# ==========================================
# 核心数据集类
# ==========================================

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 slide_label_dict={},
                 case_label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col='label',
                 patient_voting='max'):

        self.slide_label_dict = slide_label_dict
        self.case_label_dict = case_label_dict
        self.slide_num_classes = len(set(self.slide_label_dict.values()))
        self.case_num_classes = len(set(self.case_label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.slide_label_dict, self.case_label_dict, ignore)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data
        self.cls_ids_prep()

    def cls_ids_prep(self):
        # 准备 slide 级和 case 级的类别索引
        self.patient_cls_ids = [np.where(self.slide_data['case_label'] == i)[0] for i in range(self.case_num_classes)]
        self.slide_cls_ids = [np.where(self.slide_data['slide_label'] == i)[0] for i in range(self.slide_num_classes)]

    @staticmethod
    def df_prep(data, label_dict, case_label_dict, ignore):
        """将字符串标签映射为数字"""
        mask = data['slide_label'].isin(ignore)
        data = data[~mask].reset_index(drop=True)

        # 映射 slide label
        data['slide_label'] = data['slide_label'].map(label_dict)
        # 映射 case label
        data['case_label'] = data['case_label'].map(case_label_dict)

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        return len(self.slide_data)

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            # 基于内部 ID 切分 (如交叉验证生成器)
            splits = []
            for ids in [self.train_ids, self.val_ids, self.test_ids]:
                if ids is not None and len(ids) > 0:
                    split_data = self.slide_data.loc[ids].reset_index(drop=True)
                    splits.append(Generic_Split(split_data, data_dir=self.data_dir, num_classes=self.slide_num_classes))
                else:
                    splits.append(None)
            return tuple(splits)
        else:
            # 基于预定义的 CSV 切分
            assert csv_path, "Must provide csv_path for fixed splits"
            all_splits = pd.read_csv(csv_path)
            splits = []
            for key in ['train', 'val', 'test']:
                split_ids = all_splits[key].dropna().reset_index(drop=True)
                if len(split_ids) > 0:
                    mask = self.slide_data['slide_id'].isin(split_ids.tolist())
                    df_slice = self.slide_data[mask].reset_index(drop=True)
                    splits.append(Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.slide_num_classes))
                else:
                    splits.append(None)
            return tuple(splits)

    def getlabel(self, ids):
        return self.slide_data['slide_label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = str(self.slide_data['slide_id'][idx])
        case_id = str(self.slide_data['case_id'][idx])
        label = self.slide_data['slide_label'][idx]
        case_label = self.slide_data['case_label'][idx]

        data_dir = self.data_dir

        # --- 模式 1: H5 加载 (旧模式) ---
        if self.use_h5:
            full_path = os.path.join(data_dir, 'h5_files', f'{slide_id}.h5')
            with h5py.File(full_path, 'r') as hdf5_file:
                features = torch.from_numpy(hdf5_file['features'][:])
                coords = torch.from_numpy(hdf5_file['coords'][:])
            return features, label, coords, slide_id, case_label

        # --- 模式 2: PT 文件加载 (增强搜索逻辑) ---
        # 构造搜索候选词: [全名, 去前缀名, Case名]
        search_candidates = [slide_id]
        if '_' in slide_id:
            search_candidates.append(slide_id.split('_', 1)[-1])  # 去除 CD_IL_
        if case_id not in search_candidates:
            search_candidates.append(case_id)
        if '-' in slide_id:
            search_candidates.append(slide_id.rsplit('-', 1)[0])  # 去除 -1

        pt_path = None

        # 策略 A: 搜索文件夹 (适配 CORAL 结构)
        for cand in search_candidates:
            folder_pattern = os.path.join(data_dir, cand + "*")
            matched_folders = [f for f in glob.glob(folder_pattern) if os.path.isdir(f)]
            if matched_folders:
                matched_folders.sort()
                target_folder = matched_folders[0]
                # 优先级: image_feature.pt -> {id}.pt -> *.pt
                for fname in ['image_feature.pt', f"{slide_id}.pt"]:
                    if os.path.exists(os.path.join(target_folder, fname)):
                        pt_path = os.path.join(target_folder, fname)
                        break
                if pt_path is None:  # 找任意 pt
                    pts = glob.glob(os.path.join(target_folder, "*.pt"))
                    if pts: pt_path = pts[0]
                if pt_path: break

        # 策略 B: 根目录直接搜索
        if pt_path is None:
            for cand in search_candidates:
                potential = os.path.join(data_dir, f"{cand}.pt")
                if os.path.exists(potential):
                    pt_path = potential
                    break

        if pt_path is None:
            raise FileNotFoundError(f"特征文件丢失: ID={slide_id}, 搜索路径={data_dir}")

        try:
            data = torch.load(pt_path)
            # 兼容字典格式和纯 Tensor 格式
            if isinstance(data, dict):
                features = data['features']
                coords = data.get('coords', torch.zeros((features.shape[0], 2)))
            else:
                features = data
                coords = torch.zeros((features.shape[0], 2))

            if features.ndim == 1: features = features.unsqueeze(0)

            return features, label, coords, slide_id, case_label

        except Exception as e:
            raise RuntimeError(f"文件损坏或无法读取: {pt_path} | Error: {e}")


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [np.where(self.slide_data['slide_label'] == i)[0] for i in range(self.num_classes)]

    def __len__(self):
        return len(self.slide_data)


def build_dataset(cfg, datalist):
    """
    工厂函数：
    1. 实例化 Dataset
    2. 处理 ID 映射
    3. 构建 DataLoader 并计算 【每个类别的】 train_num_list
    """
    # 1. 准备标签字典 (保持不变)
    default_slide_dict = {'CD': 1, 'ITB': 0, 'TB': 0, 'NO': 2}
    default_case_dict = {'CD': 1, 'ITB': 0, 'TB': 0}

    slide_dict = getattr(cfg.DATA, 'slide_label_dict', default_slide_dict)
    if not isinstance(slide_dict, dict): slide_dict = dict(slide_dict)

    case_dict = getattr(cfg.DATA, 'case_label_dict', default_case_dict)
    if not isinstance(case_dict, dict): case_dict = dict(case_dict)

    # 2. 实例化 Dataset (保持不变)
    dataset = Generic_MIL_Dataset(
        data_dir=cfg.DATA.pt_path,
        csv_path=cfg.DATA.label_csv_file,
        slide_label_dict=slide_dict,
        case_label_dict=case_dict,
        label_col=getattr(cfg.DATA, 'label_col', 'label'),
        ignore=getattr(cfg.DATA, 'ignore', []),
        filter_dict=getattr(cfg.DATA, 'filter_dict', {}),
        seed=cfg.seed,
        print_info=True
    )

    dataset.load_from_h5(False)

    # 3. 将 Slide ID 转换为 DataFrame Index (保持不变)
    if hasattr(dataset, 'slide_data'):
        df = dataset.slide_data
        if df['slide_label'].dtype == float:
            valid_mask = df['slide_label'].notna()
            dataset.slide_data = df[valid_mask].reset_index(drop=True)
            dataset.slide_data['slide_label'] = dataset.slide_data['slide_label'].astype(int)
            df = dataset.slide_data

        id_to_idx = dict(zip(df['slide_id'].astype(str).values, df.index))

        def ids_to_indices(id_list):
            indices = []
            for split_id in id_list:
                s_id = str(split_id)
                if s_id in id_to_idx:
                    indices.append(id_to_idx[s_id])
            return indices

        if 'train' in datalist: dataset.train_ids = ids_to_indices(datalist['train'])
        if 'eval' in datalist:  dataset.val_ids = ids_to_indices(datalist['eval'])
        if 'test' in datalist:  dataset.test_ids = ids_to_indices(datalist['test'])

    # 4. 执行切分 (保持不变)
    train_d, val_d, test_d = dataset.return_splits(from_id=True)

    # =================================================================
    # 5. 【修复】 构建 DataLoader 并统计 <各类别> 数量
    # =================================================================
    loaders = {}
    num_workers = getattr(cfg.DATA, 'num_workers', 4)
    batch_size = getattr(cfg.DATA, 'batch_size', 1)
    n_classes = getattr(cfg.DATA, 'n_classes', 3)  # 获取类别总数

    # 构建 Train Loader
    if train_d is not None and len(train_d) > 0:
        loaders['train'] = get_split_loader(train_d, mode='train', num_workers=num_workers, batch_size=batch_size)

        # --- 核心修复开始 ---
        # 利用 dataset.train_ids 从原始 DataFrame 中提取训练集数据
        train_indices = dataset.train_ids
        train_df = dataset.slide_data.iloc[train_indices]

        # 统计 slide_label 列中每个类别的数量
        # value_counts() 返回如 {0: 327, 1: 91, 2: 83}
        counts_series = train_df['slide_label'].value_counts()

        # 按顺序构造列表 [count_class_0, count_class_1, count_class_2]
        train_num_list = []
        for i in range(n_classes):
            # get(i, 0) 确保如果某个类别数量为0也不会报错
            train_num_list.append(counts_series.get(i, 0))

        loaders['train_num_list'] = train_num_list
        # --- 核心修复结束 ---

    else:
        loaders['train_num_list'] = [0] * n_classes

    # 构建 Val Loader
    if val_d is not None and len(val_d) > 0:
        loaders['eval'] = get_split_loader(val_d, mode='eval', num_workers=num_workers, batch_size=batch_size)

    # 构建 Test Loader
    if test_d is not None and len(test_d) > 0:
        loaders['test'] = get_split_loader(test_d, mode='test', num_workers=num_workers, batch_size=batch_size)

    return loaders

# ==========================================
# Loader 构建函数
# ==========================================

def get_split_loader(split_dataset, mode='train', num_workers=2, batch_size=1):
    """通用 DataLoader 构建器"""
    if mode == 'train':
        weights = make_weights_for_balanced_classes_split(split_dataset)
        loader = DataLoader(split_dataset, batch_size=batch_size,
                            sampler=WeightedRandomSampler(weights, len(weights)),
                            collate_fn=collate_MIL, num_workers=num_workers)
    else:
        loader = DataLoader(split_dataset, batch_size=batch_size,
                            sampler=SequentialSampler(split_dataset),
                            collate_fn=collate_MIL, num_workers=num_workers)
    return loader


def get_loader_biospy(image_root, pt_root, split_root, fold, num_workers=2, mode='train',
                      slide_label_dict=None, case_label_dict=None, label_col=None):
    """
    主加载函数: 适用于 Biospy, Camelyon 等所有项目
    """
    dataset = Generic_MIL_Dataset(csv_path=image_root, data_dir=pt_root,
                                  slide_label_dict=slide_label_dict,
                                  case_label_dict=case_label_dict,
                                  label_col=label_col)
    dataset.load_from_h5(False)  # 强制使用 .pt

    # 获取分割
    train_d, val_d, test_d = dataset.return_splits(from_id=False,
                                                   csv_path=f'{split_root}/splits_{fold}.csv')

    # 自动处理空 Test 集
    if test_d is None:
        import pandas as pd
        test_d = Generic_Split(pd.DataFrame(columns=dataset.slide_data.columns),
                               data_dir=pt_root, num_classes=len(slide_label_dict))

    datasets = (train_d, val_d, test_d)
    stats = save_splits_stats(datasets)  # 统计数量

    # 根据模式返回 Loader
    if mode == 'train':
        loader = get_split_loader(train_d, 'train', num_workers)
    elif mode == 'eval':
        loader = get_split_loader(val_d, 'eval', num_workers)
    else:
        loader = get_split_loader(test_d, 'test', num_workers)

    return loader, stats


# 保持接口兼容性
get_loader_camelyon17 = get_loader_biospy
