# face_scanner views.py
import datetime
import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import face_recognition
from datetime import datetime
from django.conf import settings
from django.shortcuts import redirect, render
from django.http import StreamingHttpResponse
from django.contrib.auth.decorators import login_required




# #_________________________________________________________________________________

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

# # _________________________________________________________________________________

# @login_required
# def recognize_face_with_video(request):
#     user = request.user
#
#     # Load saved encoding
#     if not user.face_data_file:
#         return render(request, 'face_scanner/error.html', {'error': "No face data found. Please enroll first."})
#
#     filename = f"{user.national_code}_{user.last_name}.pkl"
#     file_path = settings.MEDIA_ROOT / "face_data" / filename
#     if not os.path.exists(file_path):
#         return render(request, 'face_scanner/error.html', {'error': "Face data file is missing."})
#
#     with open(file_path, "rb") as f:
#         saved_encoding = pickle.load(f)
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
#     start_time = time.time()
#     match_start_time = None
#     recognized = False
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#         for face_encoding in face_encodings:
#             distances = face_recognition.face_distance([saved_encoding], face_encoding)
#             if distances[0] < 0.5:  # Threshold
#                 recognized = True
#                 match_start_time = time.time() if match_start_time is None else match_start_time
#                 break
#
#         if recognized and time.time() - match_start_time >= 1.5:  # Confirm recognition for 3 seconds
#             cap.release()
#             return render(request, 'face_scanner/success_scan_with_video.html', {'message': "Face recognized successfully!"})
#
#         if time.time() - start_time > 30:  # Timeout
#             break
#
#
#
#     cap.release()
#     return render(request, 'face_scanner/error.html', {'error': "Face recognition failed or timed out."})


# #_________________________________________________________________________________
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

# #_________________________________________________________________________________
@login_required
def face_mesh_view(request):
    return render(request, 'face_scanner/face_mesh.html')


# #_________________________________________________________________________________
@login_required
def face_mesh_view_load(request):
    return render(request, 'face_scanner/face_mesh_load.html')



from django.http import HttpResponse
from .tasks import count_numbers

def start_count(request):
    count_numbers.delay()  # اجرای وظیفه به صورت غیرهمزمان
    return HttpResponse("شمارش اعداد آغاز شد!")


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

    # Initialize Face Mesh and video capture
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
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        captured_frame = frame.copy()  # Save the last captured frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance([saved_encoding], face_encoding)
            if distances[0] < 0.5:  # Threshold
                recognized = True
                match_start_time = time.time() if match_start_time is None else match_start_time
                break

        if recognized and time.time() - match_start_time >= 1.5:  # Confirm recognition for 1.5 seconds
            break

        if time.time() - start_time > 30:  # Timeout
            break

    cap.release()

    # Save the captured frame
    if captured_frame is not None:
        save_path = settings.MEDIA_ROOT / "recognized_faces"
        os.makedirs(save_path, exist_ok=True)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if recognized:
            image_name = f"{user.username}_recognized_{timestamp}.jpg"
            cv2.imwrite(str(save_path / image_name), captured_frame)
            return render(request, 'face_scanner/success_scan_with_video.html', {'message': "Face recognized successfully!"})
        else:
            save_path = settings.MEDIA_ROOT / "unrecognized_faces"
            os.makedirs(save_path, exist_ok=True)
            image_name = f"{user.username}_unrecognized_{timestamp}.jpg"
            cv2.imwrite(str(save_path / image_name), captured_frame)
            return render(request, 'face_scanner/error.html', {'error': "Face recognition failed or timed out."})
    else:
        return render(request, 'face_scanner/error.html', {'error': "No frame captured during the process."})
