from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import pickle

db = SQLAlchemy()


class Person(db.Model):
    __tablename__ = 'persons'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False, unique=True, comment='姓名')
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')

    face_embeddings = db.relationship(
        'FaceEmbedding', backref='person', lazy=True,
        cascade='all, delete-orphan'
    )
    attendance_records = db.relationship(
        'AttendanceRecord', backref='person', lazy=True,
        cascade='all, delete-orphan'
    )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        }


class FaceEmbedding(db.Model):
    __tablename__ = 'face_embeddings'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    person_id = db.Column(
        db.Integer, db.ForeignKey('persons.id'), nullable=False
    )
    embedding = db.Column(db.LargeBinary, nullable=False, comment='128维特征向量(pickle)')
    image_path = db.Column(db.String(255), comment='录入照片路径')
    created_at = db.Column(db.DateTime, default=datetime.now)

    def get_vector(self):
        return pickle.loads(self.embedding)

    def set_vector(self, arr):
        self.embedding = pickle.dumps(arr)


class AttendanceRecord(db.Model):
    __tablename__ = 'attendance_records'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    person_id = db.Column(
        db.Integer, db.ForeignKey('persons.id'), nullable=False
    )
    check_in_time = db.Column(db.DateTime, default=datetime.now, comment='签到时间')
    date = db.Column(db.Date, default=date.today, comment='签到日期')

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'person_name': self.person.name if self.person else None,
            'check_in_time': self.check_in_time.strftime('%Y-%m-%d %H:%M:%S'),
            'date': self.date.strftime('%Y-%m-%d'),
        }
