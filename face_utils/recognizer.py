import numpy as np
import pickle

from config import FACE_MATCH_THRESHOLD
from models import FaceEmbedding


class FaceRecognizer:
    """人脸识别器：使用余弦相似度进行特征比对"""

    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else FACE_MATCH_THRESHOLD

    @staticmethod
    def cosine_similarity(v1, v2):
        """计算两个向量的余弦相似度"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        return float(np.dot(v1_norm, v2_norm))

    def recognize(self, query_embedding, session):
        """比对人脸，返回最佳匹配

        Args:
            query_embedding: 查询人脸的128维向量
            session: SQLAlchemy session

        Returns:
            dict: {'person_id': int, 'person_name': str, 'similarity': float}
            或 None（未匹配到任何人）
        """
        if query_embedding is None:
            return None

        records = session.query(FaceEmbedding).all()
        if not records:
            return None

        best_match = None
        best_score = -1

        for fe in records:
            stored_vec = pickle.loads(fe.embedding)
            score = self.cosine_similarity(query_embedding, stored_vec)

            if score > best_score:
                best_score = score
                best_match = fe

        if best_match is None or best_score < self.threshold:
            return None

        return {
            'person_id': best_match.person_id,
            'person_name': best_match.person.name,
            'similarity': round(best_score, 4),
        }
