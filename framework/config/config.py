from yacs.config import CfgNode as CN


_C = CN()
_C.Task_name = "baseline"
_C.model_type = "MambaMIL"
_C.model_weights = None
_C.mode = "train"
_C.gpu_id = 0
_C.DataParallel = False
_C.logs_dir = None
_C.model_dir = None
_C.save_detail_result = True
_C.seed = 421

_C.DATA = CN()
_C.DATA.type = None
_C.DATA.batch_size = 32
_C.DATA.fold_K = 5
_C.DATA.fold_idx = 1
_C.DATA.num_workers = 8
_C.DATA.split_fold = None
_C.DATA.label_csv_file = None
_C.DATA.pt_path = None
_C.DATA.n_classes = 3
_C.DATA.case_classes = 2
_C.DATA.hidden_dim = 1024
_C.DATA.slide_label_dict = CN()
_C.DATA.slide_label_dict.CD = 1
_C.DATA.slide_label_dict.ITB = 0
_C.DATA.slide_label_dict.TB = 0
_C.DATA.slide_label_dict.NO = 2
_C.DATA.case_label_dict = CN()
_C.DATA.case_label_dict.CD = 1
_C.DATA.case_label_dict.ITB = 0
_C.DATA.case_label_dict.TB = 0
_C.DATA.label_col = "slide_label"

_C.SOLVER = CN()
_C.SOLVER.num_epochs = 200
_C.SOLVER.lr = 1e-5
_C.SOLVER.case_lr = 1e-5
_C.SOLVER.beta1 = 0.5
_C.SOLVER.beta2 = 0.999
_C.SOLVER.weight_decay = 0.0001
_C.SOLVER.momentum = 0.9
_C.SOLVER.val_step = 10
_C.SOLVER.num_epochs_decay = 100
_C.SOLVER.decay_ratio = 0.1
_C.SOLVER.lr_low = None
_C.SOLVER.lr_warm_epoch = 10
_C.SOLVER.lr_cos_epoch = 100
_C.SOLVER.activate_func = "sigmoid"
_C.SOLVER.case_activate_func = "sigmoid"
_C.SOLVER.feat_distillation = None
_C.SOLVER.METRIC = "BCEloss"
_C.SOLVER.FocalLoss = CN()
_C.SOLVER.FocalLoss.alpha = None
_C.SOLVER.FocalLoss.gamma = 4
_C.SOLVER.FocalLoss.reduction = "mean"
_C.SOLVER.LDAM = CN()
_C.SOLVER.LDAM.max_m = 0.5
_C.SOLVER.LDAM.s = 30

_C.MODEL = CN()
_C.MODEL.FRMIL = CN()
_C.MODEL.FRMIL.num_heads = 1
_C.MODEL.KSRP = CN()
_C.MODEL.KSRP.w = 0.5
_C.MODEL.KSRP.n_mu = 8
_C.MODEL.KSRP.em_iter = 2
_C.MODEL.KSRP.dropout = 0.6
_C.MODEL.KSRP.scale_factor = 10
_C.MODEL.KSRP.s_factor = 4.0
_C.MODEL.KSRP.T = 0.1
_C.MODEL.KSRP.lambda_a = 0.1
_C.MODEL.KSRP.lambda_s = 0.1
_C.MODEL.KSRP.warmup_epoch = 30
_C.MODEL.KSRP.mu_queue_len = 5
_C.MODEL.KSRP.out_feat_num = 1024

_C.MODEL.Vit = CN()
_C.MODEL.Vit.num_heads = 8
_C.MODEL.Vit.mlp_ratio = 8.0
_C.MODEL.Vit.qkv_bias = False
_C.MODEL.Vit.drop_rate = 0.6
_C.MODEL.Vit.attn_drop_rate = 0.0
_C.MODEL.Vit.depth = 6

_C.MODEL.GatedAttention = CN()
_C.MODEL.GatedAttention.L = 512
_C.MODEL.GatedAttention.D = 128
_C.MODEL.GatedAttention.K = 1

_C.MODEL.Mamba = CN()
_C.MODEL.Mamba.type = "SRMamba"
_C.MODEL.Mamba.layer = 2
_C.MODEL.Mamba.act = "gelu"
_C.MODEL.Mamba.dropout = 0.2
_C.MODEL.Mamba.rate = 10

_C.MODEL.TransMIL = CN()
_C.MODEL.TransMIL.dim = 512
