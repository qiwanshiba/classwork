import cv2
import numpy as np


class FaceDetector:
    """基于 OpenCV 的人脸检测器"""

    def __init__(self):
        # Haar 级联分类器（轻量级，速度快）
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError('Failed to load Haar cascade classifier')

    def detect(self, image):
        """检测图中所有人脸

        Args:
            image: BGR numpy array

        Returns:
            list[dict]: 每张人脸包含 'face' (crop) 和 'bbox' (x,y,w,h)
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化提升对比度
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results = []
        for (x, y, w, h) in faces:
            # 扩大检测框 10% 以获得更完整的人脸
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)

            face_img = image[y1:y2, x1:x2]
            results.append({
                'face': face_img,
                'bbox': (x, y, w, h),
            })

        return results

    def detect_largest(self, image):
        """返回图中最大的人脸"""
        results = self.detect(image)
        if not results:
            return None
        return max(results, key=lambda r: r['bbox'][2] * r['bbox'][3])
