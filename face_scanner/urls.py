from django.urls import path , include
from . import views
from django.conf import settings
from django.conf.urls.static import static

from .views import scan_and_store_face

urlpatterns = [
    # path('scan-page/', scan_page, name='scan_page'),
    path('scan/', scan_and_store_face, name='start_scan'),
    # path('save-face-data/', save_face_data, name='save_face_data'),
    path('accounts/', include('accounts.urls')),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)