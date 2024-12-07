# face_scanner urls.py

from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

from .views import *

urlpatterns = [

    path('scan/', scan_and_store_face, name='start_scan'),
    path('scan_with_video/', recognize_face_with_video, name='scan_with_video'),
    #path('scan_with_photo/', recognize_face_with_photo, name='scan_with_photo'),
    path('face-mesh/', face_mesh_view, name='face_mesh_view'),

    path('face-mesh_load/', face_mesh_view_load, name='face_mesh_view_load'),

    path('face-mesh/stream/', face_mesh_stream, name='face_mesh_stream'),
    path('accounts/', include('accounts.urls')),
    path('start-count/', views.start_count, name='start_count'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
