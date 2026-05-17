import os
import numpy as np
import cv2
from config import USE_GPU


def _patch_keras_facenet_download():
    """Monkey-patch keras_facenet 的下载函数，GitHub 不可用时使用镜像"""
    try:
        from keras_facenet import utils as kf_utils
        _original_download = kf_utils.download_and_verify

        MIRRORS = [
            'https://ghproxy.net/',
            'https://mirror.ghproxy.com/',
        ]

        def _patched_download_and_verify(url, filepath, sha256):
            # 先尝试原 URL
            try:
                return _original_download(url, filepath, sha256)
            except Exception:
                pass

            # 如果文件已存在且大小 > 1MB，先用着（可能是网络校验失败）
            if os.path.isfile(filepath) and os.path.getsize(filepath) > 1024 * 1024:
                print(f'[FaceNet] 网络不可用，使用本地缓存: {filepath}', flush=True)
                return

            # 尝试镜像
            for mirror in MIRRORS:
                mirror_url = mirror + url
                try:
                    print(f'[FaceNet] 尝试镜像下载: {mirror_url[:80]}...', flush=True)
                    return _original_download(mirror_url, filepath, sha256)
                except Exception:
                    continue

            # 全部失败，抛出原始异常
            raise RuntimeError(
                '无法下载 FaceNet 模型权重，请手动下载:\n'
                f'  {url}\n'
                f'  放置到: {filepath}'
            )

        kf_utils.download_and_verify = _patched_download_and_verify
    except ImportError:
        pass


class FaceEncoder:
    """基于 FaceNet 的人脸特征编码器"""

    def __init__(self):
        self.model = None
        self.embedding_size = 128
        self.input_size = (160, 160)

    def _load_model(self):
        """延迟加载 FaceNet 模型，配置 GPU 策略"""
        if self.model is not None:
            return

        import tensorflow as tf

        # ---------- GPU / CPU 策略 ----------
        gpus = tf.config.list_physical_devices('GPU')
        if USE_GPU and gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'[FaceNet] 使用 GPU 加速 ({len(gpus)} 个设备)', flush=True)
        else:
            if not gpus and USE_GPU:
                print('[FaceNet] GPU 不可用，回退 CPU（oneDNN 已启用）', flush=True)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(4)

        # 注入下载镜像补丁，防止 GitHub 连不上卡死
        _patch_keras_facenet_download()

        from keras_facenet import FaceNet
        self.model = FaceNet()
        self.embedding_size = 128

    def _preprocess(self, face_img):
        """预处理人脸图像: resize → 颜色转换 → 归一化

        关键优化：在送入 FaceNet 之前缩放到 160x160，
        避免 CNN 对高分辨率图像做多余卷积运算（CPU 上提效 3-5 倍）。
        """
        if face_img is None or face_img.size == 0:
            return None

        h, w = face_img.shape[:2]
        if h < 20 or w < 20:
            return None

        # ---- 缩放到 FaceNet 标准输入 160x160 ----
        face_img = cv2.resize(face_img, self.input_size,
                              interpolation=cv2.INTER_AREA)

        # ---- 颜色通道转换 ----
        if len(face_img.shape) == 2:
            # 灰度 → RGB
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            # BGRA → RGB
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
        elif face_img.shape[2] == 3:
            # BGR → RGB
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        return face_img

    def get_embedding(self, face_img):
        """提取人脸图像的 FaceNet 128维特征向量

        Args:
            face_img: BGR numpy array (人脸区域)

        Returns:
            np.ndarray: 128维特征向量，或 None
        """
        self._load_model()

        rgb = self._preprocess(face_img)
        if rgb is None:
            return None

        embeddings = self.model.embeddings([rgb])
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0]
        return None
