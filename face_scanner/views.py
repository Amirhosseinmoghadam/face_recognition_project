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
from datetime import time

#_________________________________________________________________________________

import cv2
import time
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
# @login_required
# def recognize_face_with_video(request):
#     user = request.user
#     if not user.face_data_file:
#         return render(request, 'face_scanner/error.html', {'error': "No face data found. Please complete enrollment first."})
#
#     # Load the saved encoding
#     try:
#         filename = f"{user.national_code}_{user.last_name}.pkl"
#         file_path = settings.MEDIA_ROOT / "face_data" / filename
#
#         # file_path = settings.MEDIA_ROOT / user.face_data_file.name
#         with open(file_path, "rb") as f:
#             saved_encoding = pickle.load(f)
#     except FileNotFoundError:
#         return render(request, 'face_scanner/error.html', {'error': "Face data file is missing or corrupted."})
#
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5
#     )
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot access webcam.")
#
#     match_start_time = None
#     start_time = time.time()
#     matched = False
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Recognize face
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#         for face_encoding, face_location in zip(face_encodings, face_locations):
#             distance = face_recognition.face_distance([saved_encoding], face_encoding)[0]
#
#             if distance < 0.3:  # Threshold for matching
#                 matched = True
#                 if match_start_time is None:
#                     match_start_time = time.time()
#
#                 top, right, bottom, left = face_location
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                 cv2.putText(frame, "Matched", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             else:
#                 top, right, bottom, left = face_location
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#         cv2.imshow("Face Recognition", frame)
#
#         if matched and time.time() - match_start_time >= 5:
#             cap.release()
#             cv2.destroyAllWindows()
#             return render(request, 'face_scanner/success_scan_with_video.html', {'message': "Match found!"})
#
#         if time.time() - start_time > 30 or cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     return render(request, 'face_scanner/error.html', {'error': "No match found within the time limit."})


# @login_required
# def recognize_face_with_photo(request):
#     user = request.user
#     if not user.face_data_file:
#         return render(request, 'face_scanner/error.html', {'error': "No face data found. Please complete enrollment first."})
#
#     # Load the saved encoding
#     try:
#         filename = f"{user.national_code}_{user.last_name}.pkl"
#         file_path = settings.MEDIA_ROOT / "face_data" / filename
#         # file_path = settings.MEDIA_ROOT / user.face_data_file.name
#         with open(file_path, "rb") as f:
#             saved_encoding = pickle.load(f)
#     except FileNotFoundError:
#         return render(request, 'face_scanner/error.html', {'error': "Face data file is missing or corrupted."})
#
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5
#     )
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot access webcam.")
#
#     photo_captured = False
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process the frame with Face Mesh
#         mesh_results = face_mesh.process(rgb_frame)
#         if mesh_results.multi_face_landmarks:
#             for face_landmarks in mesh_results.multi_face_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION
#                 )
#
#         cv2.imshow("Face Recognition", frame)
#
#         # Capture photo
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             photo_captured = True
#             face_locations = face_recognition.face_locations(rgb_frame)
#             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#             for face_encoding, face_location in zip(face_encodings, face_locations):
#                 distance = face_recognition.face_distance([saved_encoding], face_encoding)[0]
#
#                 if distance < 0.3:  # Threshold for matching
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     return render(request, 'face_scanner/success_scan_with_photo.html', {'message': "Match found!"})
#
#             break
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     if photo_captured:
#         return render(request, 'face_scanner/error.html', {'error': "No match found for the captured photo."})
#     return redirect('face_recognition')


import cv2
import mediapipe as mp
import face_recognition
import pickle
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.decorators import login_required
import os

@login_required
def recognize_face_with_video(request):
    user = request.user

    # Load saved encoding
    if not user.face_data_file:
        return render(request, 'face_scanner/error.html', {'error': "No face data found. Please enroll first."})

    filename = f"{user.national_code}_{user.last_name}.pkl"
    file_path = settings.MEDIA_ROOT / "face_data" / filename
    if not os.path.exists(file_path):
        return render(request, 'face_scanner/error.html', {'error': "Face data file is missing."})

    with open(file_path, "rb") as f:
        saved_encoding = pickle.load(f)

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

    start_time = time.time()
    match_start_time = None
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance([saved_encoding], face_encoding)
            if distances[0] < 0.3:  # Threshold
                recognized = True
                match_start_time = time.time() if match_start_time is None else match_start_time
                break

        if recognized and time.time() - match_start_time >= 3:  # Confirm recognition for 3 seconds
            cap.release()
            return render(request, 'face_scanner/success_scan_with_video.html', {'message': "Face recognized successfully!"})

        if time.time() - start_time > 30:  # Timeout
            break

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit
        #     break

    cap.release()
    return render(request, 'face_scanner/error.html', {'error': "Face recognition failed or timed out."})



# @login_required
# def recognize_face_with_photo(request):
#     user = request.user
#     if not user.face_data_file:
#         return render(request, 'face_scanner/error.html', {'error': "No face data found. Please complete enrollment first."})
#
#     # Load the saved encoding
#     try:
#         file_path = settings.MEDIA_ROOT / user.face_data_file.name
#         with open(file_path, "rb") as f:
#             saved_encoding = pickle.load(f)
#     except FileNotFoundError:
#         return render(request, 'face_scanner/error.html', {'error': "Face data file is missing or corrupted."})
#
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5
#     )
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot access webcam.")
#
#     photo_captured = False
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process the frame with Face Mesh
#         mesh_results = face_mesh.process(rgb_frame)
#         if mesh_results.multi_face_landmarks:
#             for face_landmarks in mesh_results.multi_face_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION
#                 )
#
#         cv2.imshow("Face Recognition", frame)
#
#         # Capture photo
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             photo_captured = True
#             face_locations = face_recognition.face_locations(rgb_frame)
#             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#             for face_encoding, face_location in zip(face_encodings, face_locations):
#                 distance = face_recognition.face_distance([saved_encoding], face_encoding)[0]
#
#                 if distance < 0.3:  # Threshold for matching
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     return render(request, 'face_scanner/success_scan_with_photo.html', {'message': "Match found!"})
#
#             break
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     if photo_captured:
#         return render(request, 'face_scanner/error.html', {'error': "No match found for the captured photo."})
#     return redirect('face_recognition')


import cv2
import numpy as np
import mediapipe as mp
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def face_mesh_stream(request):
    def generate_frames():
        # تنظیمات MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # تبدیل فریم به RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # پردازش با Face Mesh
            results = face_mesh.process(frame_rgb)

            # اگر چهره‌ای شناسایی شد
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # رسم مش چهره
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )

            # کدگذاری فریم برای ارسال
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # ارسال فریم
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@login_required
def face_mesh_view(request):
    return render(request, 'face_scanner/face_mesh.html')



@login_required
def face_mesh_view_load(request):
    return render(request, 'face_scanner/face_mesh_load.html')