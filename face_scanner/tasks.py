# face_recognition_project/counter/tasks.py

from __future__ import absolute_import, unicode_literals
from celery import shared_task
import time

@shared_task
def count_numbers():
    for i in range(1, 11):
        print(f"عدد: {i}")
        time.sleep(1)  # توقف به مدت ۱ ثانیه بین شمارش‌ها
    return "شمارش اتمام یافت."
