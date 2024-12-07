# accounts models.py
from django.db import models
import os
from django.conf import settings


# Create your models here.
from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=15, unique=True, null=True, blank=True)
    national_code = models.CharField(max_length=10, unique=True, null=True, blank=True)
    face_data_file = models.FileField(upload_to="face_data/", null=True, blank=True)

    def __str__(self):
        return self.username

    def face_data_path(self):
        """
        مسیر ذخیره فایل‌های مربوط به داده‌های چهره کاربر.
        """
        folder_path = os.path.join(settings.MEDIA_ROOT, 'face_data', self.national_code)
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, f'{self.national_code}.pkl')



# Create this model in your models.py
class UserFaceCapture(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='face_captures/')
    capture_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Face capture for {self.user.username} at {self.capture_time}"
