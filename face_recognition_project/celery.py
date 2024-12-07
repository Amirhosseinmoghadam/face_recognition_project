# face_recognition_project/face_recognition_project/celery.py

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# تنظیم متغیر محیطی پیش‌نیاز برای Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recognition_project.settings')

app = Celery('face_recognition_project')

# بارگذاری تنظیمات از فایل settings.py با پیشوند 'CELERY_'
app.config_from_object('django.conf:settings', namespace='CELERY')

# جستجوی وظایف در تمامی اپلیکیشن‌های نصب شده
app.autodiscover_tasks()
