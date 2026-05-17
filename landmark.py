import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_path = "face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                    
                landmark_list = landmark_pb2.NormalizedLandmarkList()

                landmark_list.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z
                    ) for lm in face_landmarks
                ])

                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list =landmark_list,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing.DrawingSpec(color = (0,165,255), thickness =1)

                )

                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list =landmark_list,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_contours_style()

                )

        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Face Landmark Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()