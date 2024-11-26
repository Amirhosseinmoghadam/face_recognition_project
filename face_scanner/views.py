# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# from django.http import JsonResponse
# from django.conf import settings
# import os
# import pickle
# import numpy as np
#
# @login_required
# def face_scan(request):
#     if request.method == "POST":
#         # دریافت داده‌های انکدینگ از درخواست AJAX
#         encoding = request.POST.get('encoding', None)
#         if not encoding:
#             return JsonResponse({'error': 'No encoding received.'}, status=400)
#
#         # تبدیل داده‌ها به آرایه numpy
#         encoding_array = np.array(eval(encoding))  # تبدیل رشته JSON به آرایه
#
#         # ذخیره فایل .pkl
#         user = request.user
#         directory = os.path.join(settings.MEDIA_ROOT, 'face_encodings', user.national_code)
#         os.makedirs(directory, exist_ok=True)
#         file_path = os.path.join(directory, f"{user.last_name}.pkl")
#
#         with open(file_path, 'wb') as f:
#             pickle.dump(encoding_array, f)
#
#         # ذخیره مسیر فایل در دیتابیس
#         from .models import FaceEncoding
#         FaceEncoding.objects.update_or_create(
#             user=user,
#             defaults={'national_code': user.national_code, 'last_name': user.last_name, 'encoding_file': file_path},
#         )
#
#         return JsonResponse({'message': 'Encoding saved successfully.', 'redirect_url': '/quiz/'})
#     return render(request, 'face_scanner/face_scan.html')


#_________________________________________________________________________________

import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
from django.conf import settings
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required

@login_required
def scan_and_store_face(request):
    user = request.user
    if not user.national_code:
        return render(request, 'face_scanner/error.html', {'error': "National code is missing."})

    # تنظیمات MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    face_encodings = []
    scan_complete = False
    total_landmarks = 10000
    landmarks_scanned = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # شناسایی چهره
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # MediaPipe Face Mesh: پردازش
        mesh_results = face_mesh.process(rgb_frame)
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # بررسی و شمارش نقاط
                for i, landmark in enumerate(face_landmarks.landmark):
                    if 0 < landmark.x < 1 and 0 < landmark.y < 1:
                        landmarks_scanned += 1

                if landmarks_scanned >= total_landmarks:
                    scan_complete = True
                    if encodings:
                        face_encodings.append(encodings[0])

        # پایان اسکن
        if scan_complete:
            break

    cap.release()
    # cv2.destroyAllWindows()

    # ذخیره داده‌ها
    if face_encodings:
        avg_encoding = np.mean(face_encodings, axis=0)
        filename = f"{user.national_code}_{user.last_name}.pkl"
        file_path = settings.MEDIA_ROOT / "face_data" / filename

        with open(file_path, "wb") as f:
            pickle.dump(avg_encoding, f)

        # ذخیره فایل در مدل کاربر
        user.face_data_file.name = f"face_data/{filename}"
        user.save()

    return redirect('quiz_page')  # به صفحه کوییز هدایت شود

#_________________________________________________________________________________


# from django.http import JsonResponse
#
# @login_required
# def scan_and_store_face(request):
#     if request.method == "POST":
#         user = request.user
#         if not user.national_code:
#             return JsonResponse({"success": False, "error": "National code is missing."})
#
#         # تنظیمات MediaPipe Face Mesh
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )
#
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             return JsonResponse({"success": False, "error": "Cannot access webcam."})
#
#         face_encodings = []
#         scan_complete = False
#         total_landmarks = 10000
#         landmarks_scanned = 0
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # شناسایی چهره
#             face_locations = face_recognition.face_locations(rgb_frame)
#             encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#             # MediaPipe Face Mesh: پردازش
#             mesh_results = face_mesh.process(rgb_frame)
#             if mesh_results.multi_face_landmarks:
#                 for face_landmarks in mesh_results.multi_face_landmarks:
#                     for i, landmark in enumerate(face_landmarks.landmark):
#                         if 0 < landmark.x < 1 and 0 < landmark.y < 1:
#                             landmarks_scanned += 1
#
#                     if landmarks_scanned >= total_landmarks:
#                         scan_complete = True
#                         if encodings:
#                             face_encodings.append(encodings[0])
#
#             # پایان اسکن
#             if scan_complete:
#                 break
#
#         cap.release()
#
#         # ذخیره داده‌ها
#         if face_encodings:
#             avg_encoding = np.mean(face_encodings, axis=0)
#             filename = f"{user.national_code}_{user.last_name}.pkl"
#             file_path = settings.MEDIA_ROOT / "face_data" / filename
#
#             with open(file_path, "wb") as f:
#                 pickle.dump(avg_encoding, f)
#
#             # ذخیره فایل در مدل کاربر
#             user.face_data_file.name = f"face_data/{filename}"
#             user.save()
#
#             return JsonResponse({"success": True})
#
#         return JsonResponse({"success": False, "error": "No face detected during scanning."})
#
#     return JsonResponse({"success": False, "error": "Invalid request method."})
#
#
#
# def scan_page(request):
#     return render(request, 'face_scanner/scan.html')
#
