from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

from .views import (scan_and_store_face, recognize_face_with_video,
    #recognize_face_with_photo
                    )

urlpatterns = [

    path('scan/', scan_and_store_face, name='start_scan'),
    path('scan_with_video/', recognize_face_with_video, name='scan_with_video'),
    #path('scan_with_photo/', recognize_face_with_photo, name='scan_with_photo'),

    path('accounts/', include('accounts.urls')),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
