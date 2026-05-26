# 智能考勤人脸识别系统 — 使用文档

## 1. 项目简介

基于 FaceNet + OpenCV 的深度学习人脸识别考勤系统。支持**人脸录入、自动签到、考勤统计**三大核心功能。

- 技术栈：Flask + TensorFlow + keras-facenet + OpenCV + MySQL
- 人脸检测：OpenCV Haar 级联分类器
- 特征提取：FaceNet（InceptionResNetV1），128 维特征向量
- 特征比对：余弦相似度

---

## 2. 环境要求

| 依赖 | 版本要求 |
|---|---|
| Python | >= 3.9 |
| MySQL | >= 5.7 |
| TensorFlow | >= 2.10 |
| 浏览器 | Chrome / Edge（推荐） |

**可选：GPU 加速**

本机有 NVIDIA 显卡可安装 DirectML 插件获得 2-3 倍推理加速：

```bash
pip install tensorflow-directml
```

---

## 3. 快速开始

### 3.1 安装 Python 依赖

```bash
cd #目标文件位置
pip install -r requirements.txt
```

### 3.2 创建数据库

登录 MySQL，执行：

```sql
CREATE DATABASE IF NOT EXISTS face_attendance CHARACTER SET utf8mb4;
```

### 3.3 修改数据库配置

编辑 `config.py`，把数据库连接信息改成你的：

```python
MYSQL_USER = 'root'        # 数据库用户名
MYSQL_PASSWORD = '123456'  # 数据库密码
MYSQL_HOST = 'localhost'   # 数据库地址
MYSQL_PORT = 3306          # 数据库端口
MYSQL_DB = 'face_attendance'  # 数据库名
```

### 3.4 启动

```bash
python app.py
```

看到以下输出说明启动成功：

```
=======================================================
[GPU] 检测到 xxx（或未检测到 GPU，使用 CPU 模式）
-------------------------------------------------------
[预热] 加载 FaceNet 模型中...
[预热] FaceNet 模型加载完成
=======================================================
 * Running on http://0.0.0.0:5000
```

打开浏览器访问 **http://localhost:5000** 即可。

> 注意：`debug=True` 模式下 reloader 会启动子进程，**首次请求**会重新加载模型（约 5-10 秒），之后正常。

---

## 4. 功能说明

系统共 4 个页面：

### 4.1 人脸签到（首页）

打开摄像头拍照或上传照片，系统自动识别身份并完成签到。

- 同一人**每天只签到一次**，重复签到会提示
- 签到结果会显示匹配度百分比和处理耗时
- 未录入的人脸会提示"未识别到匹配人员"

### 4.2 人脸录入

录入新人员的姓名和人脸照片。

1. 输入姓名
2. 打开摄像头拍照或上传照片（需要清晰正面照）
3. 点击"保存录入"

> 姓名不可重复，照片中必须检测到人脸。

### 4.3 签到记录

查看所有签到记录，支持按日期筛选和分页浏览。

### 4.4 考勤统计

- **概览卡片**：总注册人数、今日签到数、今日出勤率
- **近 7 天趋势图**：折线图展示每日签到人数变化
- **个人排行**：柱状图展示签到总次数 TOP 10

---

## 5. 项目结构

```
课程设计/
├── app.py                  # Flask 主程序（路由 + 业务逻辑）
├── config.py               # 配置文件（数据库、GPU、阈值）
├── models.py               # 数据库模型（Person、FaceEmbedding、AttendanceRecord）
├── requirements.txt        # Python 依赖清单
├── face_utils/
│   ├── __init__.py
│   ├── detector.py         # 人脸检测（OpenCV Haar）
│   ├── encoder.py          # 特征提取（FaceNet）
│   └── recognizer.py       # 人脸比对（余弦相似度）
├── templates/
│   ├── base.html           # 公共模板
│   ├── index.html          # 签到页
│   ├── register.html       # 录入页
│   ├── records.html        # 记录页
│   └── statistics.html     # 统计页
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/            # 录入照片存档
├── database/               # 数据库相关
└── models/                 # 模型文件
```

---

## 6. 配置说明

`config.py` 中的可调参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | 0.6 | 人脸匹配阈值，越低越宽松（容易误识），越高越严格（可能拒识） |
| `USE_GPU` | True | 是否尝试启用 GPU 加速 |
| `ALLOWED_EXTENSIONS` | png, jpg, jpeg, bmp | 允许上传的图片格式 |
| `MAX_CONTENT_LENGTH` | 10MB | 上传文件大小上限 |

---

## 7. API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/checkin` | 上传照片签到 |
| POST | `/register-face` | 录入人脸（需带 name 和 photo） |
| GET | `/api/records?date=YYYY-MM-DD&page=1` | 查询签到记录 |
| GET | `/api/statistics` | 获取统计数据 |
| GET | `/api/persons` | 获取已注册人员列表 |
| DELETE | `/api/persons/<id>` | 删除人员及关联数据 |
| GET | `/api/gpu-status` | 查看 GPU 状态 |

---

## 8. 常见问题

**Q：启动时报 "Failed to load Haar cascade"？**

OpenCV 安装不完整，重装：

```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless
```

**Q：上传照片后提示"未检测到人脸"？**

确保照片是正面、光线充足、无人脸遮挡。系统使用 Haar 级联检测器，侧脸/暗光可能检测不到。

**Q：识别不准确怎么办？**

在 `config.py` 中调低 `FACE_MATCH_THRESHOLD`（如 0.5），或者用更清晰的照片重新录入。

**Q：第一次签到特别慢？**

这是正常现象——`debug=True` 模式下子进程需重新加载模型。后续会快。生产环境建议设 `debug=False`。

**Q：如何启用 GPU 加速？**

```bash
pip install tensorflow-directml
```

重启后查看终端输出，确认显示 "DirectML GPU 已启用"。

---

## 9. 性能参考

| 场景 | CPU 模式 | GPU（DirectML） |
|---|---|---|
| 人脸录入（首次） | 8-15 秒 | 3-5 秒 |
| 人脸录入（后续） | 3-8 秒 | 1-3 秒 |
| 签到识别 | 3-8 秒 | 1-3 秒 |

*测试环境：Intel i5 + RTX 4060，录入人数 < 50*
