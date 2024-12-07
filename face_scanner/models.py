# face scanner models.py
from django.db import models
from django.conf import settings
import os

from accounts.models import CustomUser


def user_face_directory_path(instance, filename):
    return f"media/faces/{instance.user.national_code}/{instance.user.last_name}_{filename}"

class FaceEncoding(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    encoding_file = models.FileField(upload_to=user_face_directory_path)

    def __str__(self):
        return f"{self.user}'s Face Data"
