from django.urls import path
from face_scanner import consumers

websocket_urlpatterns = [
    path('ws/face-mesh/', consumers.FaceMeshConsumer.as_asgi()),
]
