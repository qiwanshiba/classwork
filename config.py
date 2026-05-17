import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = 'face-attendance-secret-key-2024'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

# Face recognition threshold
FACE_MATCH_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# GPU 加速配置
# ---------------------------------------------------------------------------
# 设为 True 尝试启用 GPU 加速（需 NVIDIA 显卡 + CUDA + cuDNN）
# Windows 上 TensorFlow >= 2.11 不支持原生 GPU，会自动回退 CPU
# 如需 GPU 请安装: pip install tensorflow-directml
USE_GPU = True


def detect_gpu():
    """检测 GPU 可用状态，返回 (status, message)"""
    info = {
        'gpu_available': False,
        'gpu_name': None,
        'backend': 'CPU',
        'message': '',
        'suggestions': [],
    }

    # 1) 检查 NVIDIA 驱动
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info['gpu_name'] = result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    # 2) 检查 TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info['gpu_available'] = True
            info['backend'] = 'TensorFlow-GPU'
            info['message'] = f'GPU 已启用: {gpus[0].name}'
            return info
    except Exception:
        pass

    # 3) 尝试 DirectML (Windows 原生 GPU 方案)
    if not info['gpu_available'] and info['gpu_name']:
        try:
            import tensorflow as tf
            # 尝试加载 DirectML 插件
            try:
                import tensorflow_directml
                info['backend'] = 'DirectML'
                info['gpu_available'] = True
                info['message'] = f'DirectML GPU 已启用: {info["gpu_name"]}'
                return info
            except ImportError:
                info['suggestions'].append(
                    '安装 DirectML 插件以启用 GPU: pip install tensorflow-directml'
                )
        except Exception:
            pass

    # 4) CPU 回退
    if info['gpu_name']:
        info['message'] = (
            f'检测到 {info["gpu_name"]}，但 TensorFlow >= 2.11 在 Windows 上'
            f'不支持原生 GPU。将使用 CPU（已启用 oneDNN 优化）。'
        )
    else:
        info['message'] = '未检测到 GPU，使用 CPU 模式。'

    if not info['gpu_available'] and info['gpu_name']:
        info['suggestions'].extend([
            '方案A: pip install tensorflow-directml  (推荐，使用 DirectX 12)',
            '方案B: 在 WSL2 中运行本项目  (原生 CUDA 支持)',
        ])

    return info


# MySQL配置
MYSQL_USER = 'root'
MYSQL_PASSWORD = '123456'
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_DB = 'face_attendance'

SQLALCHEMY_DATABASE_URI = (
    f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@'
    f'{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4'
)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 保存上传图像的路径
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
