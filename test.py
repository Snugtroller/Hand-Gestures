import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)  # Try different indices if needed
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hand landmarks
    results = hands.process(frame_rgb)

    # If hand landmarks are found, draw them on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Display the frame with or without hand landmarks
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
