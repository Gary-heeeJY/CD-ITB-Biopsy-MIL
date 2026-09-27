import glob
import glob
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

from data.utils import make_datalist


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return dict(value)


def _load_feature_file(data_dir, slide_id, case_id):
    search_candidates = [slide_id]
    if "_" in slide_id:
        search_candidates.append(slide_id.split("_", 1)[-1])
    if case_id not in search_candidates:
        search_candidates.append(case_id)
    if "-" in slide_id:
        search_candidates.append(slide_id.rsplit("-", 1)[0])

    pt_path = None
    for cand in search_candidates:
        folder_pattern = os.path.join(data_dir, cand + "*")
        matched_folders = [f for f in glob.glob(folder_pattern) if os.path.isdir(f)]
        if not matched_folders:
            continue

        matched_folders.sort()
        target_folder = matched_folders[0]
        for fname in ("image_feature.pt", f"{slide_id}.pt"):
            candidate_path = os.path.join(target_folder, fname)
            if os.path.exists(candidate_path):
                pt_path = candidate_path
                break
        if pt_path is None:
            pt_files = glob.glob(os.path.join(target_folder, "*.pt"))
            if pt_files:
                pt_path = pt_files[0]
        if pt_path:
            break

    if pt_path is None:
        for cand in search_candidates:
            candidate_path = os.path.join(data_dir, f"{cand}.pt")
            if os.path.exists(candidate_path):
                pt_path = candidate_path
                break

    if pt_path is None:
        h5_path = os.path.join(data_dir, "h5_files", f"{slide_id}.h5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, "r") as hdf5_file:
                return torch.from_numpy(hdf5_file["features"][:])
        raise FileNotFoundError(f"Missing feature file: slide_id={slide_id}, case_id={case_id}, data_dir={data_dir}")

    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict):
        features = data["features"]
    else:
        features = data
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features.float()


class PatientMILBagDataset(Dataset):
    """
    Patient-level MIL dataset.

    Each sample is one patient bag built by concatenating all patch features from
    the patient's slides in the selected split. This is the direct patch-to-patient
    comparison setting for ABMIL, TransMIL, and MambaMIL.
    """

    def __init__(self, cfg, split_ids, phase="train", max_patches_per_patient=0):
        self.cfg = cfg
        self.phase = phase
        self.data_dir = cfg.DATA.pt_path
        self.max_patches_per_patient = int(max_patches_per_patient or 0)

        slide_label_dict = _as_dict(getattr(cfg.DATA, "slide_label_dict", {"CD": 1, "ITB": 0, "TB": 0, "NO": 2}))
        case_label_dict = _as_dict(getattr(cfg.DATA, "case_label_dict", {"CD": 1, "ITB": 0, "TB": 0}))

        slide_data = pd.read_csv(cfg.DATA.label_csv_file)
        slide_data = slide_data[slide_data["slide_label"].isin(slide_label_dict.keys())].reset_index(drop=True)
        slide_data["slide_label"] = slide_data["slide_label"].map(slide_label_dict)
        slide_data["case_label"] = slide_data["case_label"].map(case_label_dict)
        slide_data = slide_data.dropna(subset=["case_label"]).reset_index(drop=True)
        slide_data["slide_id"] = slide_data["slide_id"].astype(str)
        slide_data["case_id"] = slide_data["case_id"].astype(str)
        slide_data["case_label"] = slide_data["case_label"].astype(int)

        split_set = {str(item) for item in split_ids if not pd.isna(item)}
        split_data = slide_data[slide_data["slide_id"].isin(split_set)].copy()
        if split_data.empty:
            raise ValueError(f"No slides found for split '{phase}'. Check split CSV and label_csv_file.")

        self.patient_groups = []
        for case_id, group_df in split_data.groupby("case_id", sort=True):
            labels = group_df["case_label"].unique().tolist()
            if len(labels) != 1:
                raise ValueError(f"Patient {case_id} has inconsistent case labels: {labels}")
            self.patient_groups.append(
                {
                    "case_id": str(case_id),
                    "case_label": int(labels[0]),
                    "slides": group_df[["slide_id", "case_id"]].to_dict("records"),
                }
            )

        self.cls_ids = [
            np.where(np.array([item["case_label"] for item in self.patient_groups]) == cls_idx)[0]
            for cls_idx in range(cfg.DATA.case_classes)
        ]

    def __len__(self):
        return len(self.patient_groups)

    def __getitem__(self, idx):
        item = self.patient_groups[idx]
        features = []
        slide_ids = []
        for slide in item["slides"]:
            slide_id = str(slide["slide_id"])
            case_id = str(slide["case_id"])
            features.append(_load_feature_file(self.data_dir, slide_id, case_id))
            slide_ids.append(slide_id)

        bag = torch.cat(features, dim=0)
        if self.max_patches_per_patient > 0 and bag.shape[0] > self.max_patches_per_patient:
            generator = torch.Generator().manual_seed(self.cfg.seed + idx)
            if self.phase == "train":
                selected = torch.randperm(bag.shape[0], generator=generator)[: self.max_patches_per_patient]
            else:
                selected = torch.arange(self.max_patches_per_patient)
            bag = bag[selected]

        return bag, int(item["case_label"]), item["case_id"], slide_ids

    def getlabel(self, idx):
        return self.patient_groups[idx]["case_label"]


def patient_collate_mil(batch):
    if len(batch) != 1:
        raise ValueError("Patient-level MIL uses variable-size bags; set DATA.batch_size=1.")
    feat, label, case_id, slide_ids = batch[0]
    return feat, torch.LongTensor([label]), case_id, slide_ids


def _make_weights(dataset):
    labels = [dataset.getlabel(idx) for idx in range(len(dataset))]
    counts = np.bincount(labels, minlength=dataset.cfg.DATA.case_classes)
    weights_per_class = [len(labels) / max(count, 1) for count in counts]
    return torch.DoubleTensor([weights_per_class[label] for label in labels])


def _count_classes(dataset):
    labels = [dataset.getlabel(idx) for idx in range(len(dataset))]
    counts = np.bincount(labels, minlength=dataset.cfg.DATA.case_classes)
    return counts.astype(int).tolist()


def build_patient_mil_loaders(cfg, max_patches_per_patient=0):
    datalist = make_datalist(cfg)
    loaders = {}
    datasets = {}

    for phase in ("train", "eval", "test"):
        split_ids = datalist.get(phase, [])
        if not split_ids:
            continue
        dataset = PatientMILBagDataset(
            cfg,
            split_ids=split_ids,
            phase=phase,
            max_patches_per_patient=max_patches_per_patient,
        )
        datasets[phase] = dataset
        if phase == "train":
            sampler = WeightedRandomSampler(_make_weights(dataset), len(dataset), replacement=True)
        else:
            sampler = SequentialSampler(dataset)
        loaders[phase] = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=getattr(cfg.DATA, "num_workers", 4),
            collate_fn=patient_collate_mil,
        )

    loaders["train_num_list"] = _count_classes(datasets["train"])
    loaders["datasets"] = datasets
    return loaders
