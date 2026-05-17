import os
import io
import uuid
from datetime import datetime, date

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for
)

from config import SECRET_KEY, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, detect_gpu as check_gpu
from models import db, Person, FaceEmbedding, AttendanceRecord
from face_utils import FaceDetector, FaceEncoder, FaceRecognizer

# ---------------------------------------------------------------------------
# App 初始化
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_pyfile('config.py')
app.secret_key = SECRET_KEY

db.init_app(app)

detector = FaceDetector()
encoder = FaceEncoder()
recognizer = FaceRecognizer()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(file, max_size=800):
    """压缩图片到指定最大尺寸，返回 OpenCV BGR 图像

    优化：直接用 cv2.imdecode 从字节流解码，避免 PIL→numpy→cv2 中转。
    """
    # 读取文件字节流
    if hasattr(file, 'read'):
        file_bytes = file.read()
    else:
        file_bytes = file

    # cv2 直接从内存解码
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图片')

    h, w = img.shape[:2]
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def save_upload(file):
    """压缩并保存上传文件，返回 (保存路径, BGR numpy 数组)

    优化：同时返回内存中的 numpy 数组，避免调用方再次从磁盘读取。
    """
    ext = 'jpg'
    filename = f'{uuid.uuid4().hex}.{ext}'
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    img = compress_image(file)
    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return save_path, img


def read_image(path):
    """用 OpenCV 读取图片"""
    img = cv2.imread(path)
    if img is None:
        pil_img = Image.open(path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# 路由 - 页面
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/records')
def records_page():
    return render_template('records.html')


@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')


# ---------------------------------------------------------------------------
# 路由 - 签到
# ---------------------------------------------------------------------------
@app.route('/checkin', methods=['POST'])
def checkin():
    """上传照片进行签到（内存处理，不落盘）"""
    t_start = datetime.now()

    if 'photo' not in request.files:
        return jsonify({'success': False, 'message': '请上传照片'})

    file = request.files['photo']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '不支持的图片格式'})

    # 直接压缩并读入内存
    image = compress_image(file.stream)

    # 检测人脸
    face_data = detector.detect_largest(image)
    if face_data is None:
        return jsonify({'success': False, 'message': '未检测到人脸，请重拍'})

    # 提取特征
    embedding = encoder.get_embedding(face_data['face'])
    if embedding is None:
        return jsonify({'success': False, 'message': '人脸特征提取失败'})

    # 识别
    result = recognizer.recognize(embedding, db.session)

    if result is None:
        return jsonify({
            'success': False,
            'message': '未识别到匹配人员，请先录入人脸',
        })

    # 检查今天是否已签到
    today = date.today()
    existing = AttendanceRecord.query.filter(
        AttendanceRecord.person_id == result['person_id'],
        AttendanceRecord.date == today,
    ).first()

    if existing:
        return jsonify({
            'success': True,
            'already_checked': True,
            'person_name': result['person_name'],
            'check_in_time': existing.check_in_time.strftime('%H:%M:%S'),
            'similarity': result['similarity'],
            'elapsed': round((datetime.now() - t_start).total_seconds(), 2),
        })

    # 记录签到
    record = AttendanceRecord(
        person_id=result['person_id'],
        check_in_time=datetime.now(),
        date=today,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        'success': True,
        'already_checked': False,
        'person_name': result['person_name'],
        'check_in_time': record.check_in_time.strftime('%H:%M:%S'),
        'similarity': result['similarity'],
        'elapsed': round((datetime.now() - t_start).total_seconds(), 2),
    })


# ---------------------------------------------------------------------------
# 路由 - 人脸录入
# ---------------------------------------------------------------------------
@app.route('/register-face', methods=['POST'])
def register_face():
    """录入新人员的人脸"""
    t_start = datetime.now()

    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': '请输入姓名'})

    if 'photo' not in request.files:
        return jsonify({'success': False, 'message': '请上传照片'})

    file = request.files['photo']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '不支持的图片格式'})

    # 检查姓名是否已存在
    existing_person = Person.query.filter_by(name=name).first()
    if existing_person:
        return jsonify({'success': False, 'message': f'姓名 "{name}" 已存在'})

    # 保存图片并直接获取 numpy 数组（避免写盘后再读回）
    path, image = save_upload(file)

    # 检测人脸
    face_data = detector.detect_largest(image)
    if face_data is None:
        return jsonify({'success': False, 'message': '未检测到人脸，请上传正面照片'})

    # 提取特征
    embedding = encoder.get_embedding(face_data['face'])
    if embedding is None:
        return jsonify({'success': False, 'message': '特征提取失败，请换一张照片'})

    # 保存到数据库
    person = Person(name=name)
    db.session.add(person)
    db.session.flush()  # 获取 person.id

    fe = FaceEmbedding(person_id=person.id, image_path=path)
    fe.set_vector(embedding)
    db.session.add(fe)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'{name} 录入成功！',
        'person': person.to_dict(),
        'elapsed': round((datetime.now() - t_start).total_seconds(), 2),
    })


# ---------------------------------------------------------------------------
# 路由 - 签到记录查询
# ---------------------------------------------------------------------------
@app.route('/api/records')
def get_records():
    """获取签到记录，支持日期筛选"""
    date_str = request.args.get('date', '')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    query = AttendanceRecord.query.join(Person)

    if date_str:
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(AttendanceRecord.date == filter_date)
        except ValueError:
            pass

    query = query.order_by(AttendanceRecord.check_in_time.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    records = [r.to_dict() for r in pagination.items]

    return jsonify({
        'success': True,
        'records': records,
        'total': pagination.total,
        'pages': pagination.pages,
        'current': page,
    })


# ---------------------------------------------------------------------------
# 路由 - 统计
# ---------------------------------------------------------------------------
@app.route('/api/statistics')
def get_statistics():
    """获取考勤统计数据"""
    # 所有已注册人员
    persons = Person.query.all()
    total_persons = len(persons)

    today = date.today()

    # 今日签到
    today_count = AttendanceRecord.query.filter(
        AttendanceRecord.date == today
    ).count()

    # 近7天签到趋势
    from datetime import timedelta
    daily_data = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        count = AttendanceRecord.query.filter(
            AttendanceRecord.date == d
        ).count()
        daily_data.append({
            'date': d.strftime('%m-%d'),
            'count': count,
        })

    # 个人签到排名（总签到次数）
    from sqlalchemy import func
    top = db.session.query(
        Person.name,
        func.count(AttendanceRecord.id).label('total')
    ).join(AttendanceRecord, Person.id == AttendanceRecord.person_id)\
     .group_by(Person.id)\
     .order_by(func.count(AttendanceRecord.id).desc())\
     .limit(10)\
     .all()

    ranking = [{'name': row.name, 'total': row.total} for row in top]

    return jsonify({
        'success': True,
        'total_persons': total_persons,
        'today_checked': today_count,
        'daily_trend': daily_data,
        'ranking': ranking,
    })


# ---------------------------------------------------------------------------
# 路由 - 人员列表
# ---------------------------------------------------------------------------
@app.route('/api/persons')
def get_persons():
    """获取所有已注册人员"""
    persons = Person.query.order_by(Person.created_at.desc()).all()
    return jsonify({
        'success': True,
        'persons': [p.to_dict() for p in persons],
    })


@app.route('/api/persons/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """删除人员及其相关数据"""
    person = Person.query.get_or_404(person_id)
    db.session.delete(person)
    db.session.commit()
    return jsonify({'success': True, 'message': '已删除'})


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def _prewarm():
    """启动时检测 GPU、预加载 FaceNet 模型"""
    print('=' * 55, flush=True)
    gpu_info = check_gpu()
    print(f'[GPU] {gpu_info["message"]}', flush=True)
    for s in gpu_info.get('suggestions', []):
        print(f'[GPU] 建议: {s}', flush=True)
    print('-' * 55, flush=True)
    print('[预热] 加载 FaceNet 模型中，大约需要 30-60 秒，请耐心等待...', flush=True)
    encoder._load_model()
    print('[预热] FaceNet 模型加载完成', flush=True)
    print('=' * 55, flush=True)


# ---------------------------------------------------------------------------
# 路由 - GPU 状态查询
# ---------------------------------------------------------------------------
@app.route('/api/gpu-status')
def gpu_status():
    """返回 GPU 状态信息"""
    info = check_gpu()
    return jsonify({
        'success': True,
        'gpu_available': info['gpu_available'],
        'gpu_name': info['gpu_name'],
        'backend': info['backend'],
        'message': info['message'],
        'suggestions': info.get('suggestions', []),
    })


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    _prewarm()
    app.run(host='0.0.0.0', port=5000, debug=False)
