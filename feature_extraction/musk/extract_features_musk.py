import os
import sys
import torch
import h5py
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import openslide
import time
import json
from concurrent.futures import ThreadPoolExecutor
from config import Config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.stain import MacenkoStainNormalizer
import MUSK_model.modeling
from MUSK_model.musk_model import FusionClassifier

class FeatureExtractor:
    def __init__(self, batch_size=64, max_workers=4):
        self.device = Config.DEVICE
        self.batch_size = Config.BATCH_SIZE
        self.max_workers = Config.MAX_NUM_WORKERS  # 线程池工作线程数
        self.model = self._load_model()
        self.stain_normalizer = MacenkoStainNormalizer()
        os.makedirs(Config.FEATURE_DIR, exist_ok=True)
        print(f"特征提取器已初始化，使用设备: {self.device}")
        print(f"批处理大小: {self.batch_size}")
        print(f"并行读取线程数: {self.max_workers}")
        print(f"特征保存目录: {Config.FEATURE_DIR}")

    def _load_model(self):
        """加载MUSK模型（冻结权重）"""
        print("正在加载MUSK模型...")
        model = FusionClassifier().to(self.device)
        for param in model.musk.parameters():
            param.requires_grad = False
        model.eval()
        print("模型加载完成!")
        return model

    def extract_from_directory(self):
        """自动扫描切片目录，提取所有有效样本的图像特征（跳过已处理样本）"""
        # 1. 扫描切片目录，获取支持格式的 slide_id
        slide_exts = tuple(getattr(Config, "SLIDE_EXTENSIONS", (Config.SLIDE_EXT,)))
        h5_ext = ".h5"

        # 获取所有切片文件的文件名（不含路径）
        slide_files = [f for f in os.listdir(Config.SLIDE_DIR)
                       if f.lower().endswith(tuple(ext.lower() for ext in slide_exts))
                       and os.path.isfile(os.path.join(Config.SLIDE_DIR, f))]

        # 提取slide_id（文件名前缀，去掉后缀）
        slide_ids = [os.path.splitext(f)[0] for f in slide_files]

        # 2. 筛选出存在对应.h5文件的slide_id
        valid_slide_ids = []
        for slide_id in slide_ids:
            h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{h5_ext}")
            if os.path.exists(h5_path):
                valid_slide_ids.append(slide_id)
            else:
                print(f"警告：{slide_id}{h5_ext} 不存在，跳过该样本")

        if not valid_slide_ids:
            print("未找到有效样本（无匹配的.tif和.h5文件）")
            return

        # 3. 筛选出未处理的样本（核心新增：检查是否已存在完整输出）
        unprocessed_ids = []
        for slide_id in valid_slide_ids:
            save_dir = os.path.join(Config.FEATURE_DIR, slide_id)
            # 检查是否存在特征文件和元数据文件（两者都存在才视为已处理）
            feature_file = os.path.join(save_dir, "image_feature.pt")
            meta_file = os.path.join(save_dir, "patch_metadata.json")
            if os.path.exists(feature_file) and os.path.exists(meta_file):
                print(f"已处理：{slide_id}（跳过）")
            else:
                unprocessed_ids.append(slide_id)

        print(f"共发现 {len(valid_slide_ids)} 个有效样本，其中 {len(unprocessed_ids)} 个未处理，开始提取...")

        # 4. 仅处理未处理的样本
        main_bar = tqdm(unprocessed_ids, desc="提取图像特征", unit="样本")
        success_count = 0
        failed_count = 0

        for slide_id in main_bar:
            original_id = slide_id
            try:
                start_time = time.time()

                # 提取图像特征和元数据
                stacked_features, patch_metadata = self.extract_patch_features_optimized(slide_id)

                # 创建保存目录并保存
                save_dir = os.path.join(Config.FEATURE_DIR, original_id)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(stacked_features, os.path.join(save_dir, "image_feature.pt"))
                with open(os.path.join(save_dir, "patch_metadata.json"), "w") as f:
                    json.dump(patch_metadata, f, indent=2)
                with open(os.path.join(save_dir, "feature_extractor_metadata.json"), "w") as f:
                    json.dump({
                        "slide_id": original_id,
                        "feature_extractor": "MUSK",
                        "feature_dim": int(stacked_features.shape[1]) if stacked_features.ndim == 2 else 1024,
                        "n_patches_used": int(stacked_features.shape[0]) if stacked_features.ndim > 0 else 0,
                        "stain_normalization": "Macenko",
                        "ms_aug": False,
                    }, f, indent=2)

                elapsed = time.time() - start_time
                success_count += 1

                main_bar.set_postfix({
                    "成功": success_count,
                    "失败": failed_count,
                    "剩余": len(unprocessed_ids) - success_count - failed_count,
                    "耗时": f"{elapsed:.1f}s"
                })

            except Exception as e:
                failed_count += 1
                main_bar.write(f"处理 {slide_id} 失败: {str(e)}")
                elapsed = time.time() - start_time
                # 清理失败的空文件夹
                save_dir = os.path.join(Config.FEATURE_DIR, original_id)
                if os.path.exists(save_dir) and not os.listdir(save_dir):
                    os.rmdir(save_dir)

                main_bar.set_postfix({
                    "成功": success_count,
                    "失败": failed_count,
                    "剩余": len(unprocessed_ids) - success_count - failed_count,
                    "耗时": f"{elapsed:.1f}s"
                })

        print(f"\n{'=' * 50}")
        print(f"处理完成！成功提取 {success_count}/{len(unprocessed_ids)} 个未处理样本的特征")
        if failed_count > 0:
            print(f"失败样本数: {failed_count}")
        print(f"特征保存目录：{Config.FEATURE_DIR}")
        print(f"{'=' * 50}")

    def extract_patch_features_optimized(self, slide_id):
        """优化版本：使用线程池并行读取patch"""
        slide_exts = tuple(getattr(Config, "SLIDE_EXTENSIONS", (Config.SLIDE_EXT,)))
        h5_ext = ".h5"

        h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{h5_ext}")
        slide_path = next(
            (os.path.join(Config.SLIDE_DIR, f"{slide_id}{extension}")
             for extension in slide_exts
             if os.path.exists(os.path.join(Config.SLIDE_DIR, f"{slide_id}{extension}"))),
            None,
        )

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5文件不存在: {h5_path}")
        if slide_path is None:
            raise FileNotFoundError(f"切片文件不存在: {slide_id} with extensions {slide_exts}")

        wsi = openslide.open_slide(slide_path)

        def log_msg(msg):
            tqdm.write(f"[{slide_id}] {msg}")

        log_msg(f"开始处理，切片文件: {os.path.basename(slide_path)}")

        all_features = []
        patch_metadata = []

        with h5py.File(h5_path, "r") as h5_file:
            coords = h5_file["coords"][:]
            patch_level = int(h5_file["coords"].attrs["patch_level"])
            patch_size = int(h5_file["coords"].attrs["patch_size"])

        log_msg(f"发现 {len(coords)} 个patch（层级: {patch_level}, 尺寸: {patch_size})")

        is_half = next(self.model.musk.parameters()).dtype == torch.float16
        log_msg(f"预处理精度: {'半精度(float16)' if is_half else '单精度(float32)'}")
        log_msg(f"批处理大小: {self.batch_size}")
        log_msg(f"使用线程池加速读取")

        # 计算批次数量
        n_patches = len(coords)
        n_batches = (n_patches + self.batch_size - 1) // self.batch_size

        # 使用批次进度条
        patch_bar = tqdm(
            range(n_batches), total=n_batches,
            desc=f"{slide_id} - 处理批次",
            unit="批次", leave=False
        )

        # 创建线程池
        thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        for batch_idx in patch_bar:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_patches)

            # 提交并行读取任务
            futures = []
            batch_coords = coords[start_idx:end_idx]

            for i, coord in enumerate(batch_coords):
                future = thread_pool.submit(
                    self._read_and_preprocess_single_patch,
                    wsi, coord, patch_level, patch_size, is_half
                )
                futures.append((future, coord))

            # 收集结果
            batch_imgs = []
            for future, coord in futures:
                try:
                    img_tensor = future.result()
                    batch_imgs.append(img_tensor)

                    # 记录元数据
                    patch_metadata.append({
                        "coord": [int(coord[0]), int(coord[1])],
                        "patch_level": patch_level,
                        "patch_size": patch_size
                    })
                except Exception as e:
                    log_msg(f"读取patch失败 {coord}: {str(e)}")
                    # 添加一个空的张量占位
                    batch_imgs.append(torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE)))

            # 如果有图像，批量处理
            if batch_imgs:
                # 合并批次
                batch_tensor = torch.stack(batch_imgs).to(self.device)

                with torch.no_grad():
                    img_feats, _ = self.model.musk(
                        image=batch_tensor,
                        return_global=True,
                        with_head=False,
                        out_norm=False,
                        ms_aug=False
                    )

                # 添加到特征列表
                all_features.append(img_feats.cpu())

            # 更新进度条
            patch_bar.set_postfix(进度=f"{end_idx}/{n_patches}")

        # 关闭线程池
        thread_pool.shutdown(wait=True)
        patch_bar.close()

        log_msg(f"完成所有patch处理")

        # 合并所有批次的特征
        if all_features:
            stacked_feats = torch.cat(all_features, dim=0)
        else:
            stacked_feats = torch.tensor([])

        log_msg(f"特征堆叠完成，形状: {stacked_feats.shape}")

        # 关闭WSI
        wsi.close()

        return stacked_feats, patch_metadata

    def _read_and_preprocess_single_patch(self, wsi, coord, patch_level, patch_size, is_half):
        """读取和预处理单个patch（用于线程池）"""
        img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert("RGB")

        if Config.TARGET_PATCH_SIZE > 0:
            img = img.resize((Config.TARGET_PATCH_SIZE, Config.TARGET_PATCH_SIZE))

        img = self.stain_normalizer(img)
        img_tensor = self._preprocess_image(img, is_half)
        return img_tensor

    def _preprocess_image(self, img, is_half_precision=False):
        transform = T.Compose([
            T.Resize(Config.IMG_SIZE),
            T.CenterCrop(Config.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(img)
        if is_half_precision:
            tensor = tensor.float()
        return tensor


def main():
    print("=" * 50)
    print("===== 优化版图像特征提取工具 =====")
    print(f"切片目录: {Config.SLIDE_DIR} (后缀: {Config.SLIDE_EXT})")
    print(f"Patch目录: {Config.PATCH_DIR} (后缀: .h5)")
    print(f"特征保存目录: {Config.FEATURE_DIR}")
    print("=" * 50)

    try:
        # 创建提取器时可以指定batch_size和线程数
        extractor = FeatureExtractor(batch_size=Config.BATCH_SIZE_Extract_Feature, max_workers=4)
        extractor.extract_from_directory()
    except Exception as e:
        import traceback
        print(f"\n处理出错: {str(e)}")
        print(traceback.format_exc())
        exit(1)

    print("\n===== 所有操作完成 =====")


if __name__ == "__main__":
    main()
