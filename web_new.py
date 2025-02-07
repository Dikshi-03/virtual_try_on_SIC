import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

def main():
    # Use the default webcam (0) as the video source or a video file
    cap = cv2.VideoCapture(r'C:\Users\Dikishitha\Downloads\Virtual_try-on-main\Virtual_try-on-main\Resources\Videos\m4.mp4')  # Use your video file path

    # Initialize the PoseDetector class
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Shirt images paths
    shirt_images = [
        r"C:\Users\Dikishitha\Downloads\Virtual_try-on-main\Virtual_try-on-main\Resources\Shirts\1.png",
        r"C:\Users\Dikishitha\Downloads\Virtual_try-on-main\Virtual_try-on-main\Resources\Shirts\2.png",
        r"C:\Users\Dikishitha\Downloads\Virtual_try-on-main\Virtual_try-on-main\Resources\Shirts\3.png"
    ]
    current_shirt_index = 0  # Initial shirt index

    # Load shirt images
    for shirt_path in shirt_images:
        shirt_img = cv2.imread(shirt_path)
        if shirt_img is not None:
            print(f"Shirt image {shirt_path} loaded successfully.")
        else:
            print(f"Error loading shirt image {shirt_path}.")

    while True:
        # Capture each frame from the video
        success, img = cap.read()

        if not success:
            print("Error reading frame.")
            break

        # Find the pose in the current frame
        img = detector.findPose(img)

        # Find landmarks and bounding box information
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

        # Check if any landmarks are detected
        if lmList:
            # Get the position of the left and right hands (landmarks 15, 16)
            left_wrist = lmList[14]
            right_wrist = lmList[15]

            # Get the position of the shoulders (landmarks 11, 12)
            left_shoulder = lmList[10]
            right_shoulder = lmList[11]

            # Detect gestures: hand above the head
            if left_wrist[1] < left_shoulder[1] - 50:  # Left hand above head
                current_shirt_index = (current_shirt_index - 1) % len(shirt_images)  # Cycle left
            elif right_wrist[1] < right_shoulder[1] - 50:  # Right hand above head
                current_shirt_index = (current_shirt_index + 1) % len(shirt_images)  # Cycle right

            # Load the shirt image based on the current shirt index
            shirt_img = cv2.imread(shirt_images[current_shirt_index])

            # Define a more reasonable scaling factor to fit within torso
            torso_height = lmList[23][1] - lmList[11][1]  # Distance between shoulders and waist (landmarks 11 and 23)
            shirt_width = int(torso_height * 0.8)  # Adjust width based on height

            # Resize the shirt image to fit within the torso dimensions
            shirt_img = cv2.resize(shirt_img, (shirt_width, torso_height))

            # Define the shirt's position on the torso (roughly the region between shoulders and waist)
            shirt_y = lmList[0][2]  # The vertical position (shoulder height)
            shirt_x = lmList[5][0]  # The horizontal position (shoulder width)

            # Ensure the shirt image fits within the frame boundaries
            if shirt_y + shirt_img.shape[0] > img.shape[0]:
                shirt_img = cv2.resize(shirt_img, (shirt_width, img.shape[0] - shirt_y))

            # Correct the placement of the shirt by ensuring it is centered
            shirt_x = int(left_shoulder[0] - shirt_img.shape[1] // 2)  # Centering the shirt on the torso
            shirt_y = int(left_shoulder[1])  # Starting from shoulder height

            # Place the resized shirt image onto the frame
            try:
                img[shirt_y:shirt_y + shirt_img.shape[0], shirt_x:shirt_x + shirt_img.shape[1]] = shirt_img
            except ValueError as e:
                print(f"Error: {e}")
                continue

            # Optionally, you can show a rectangle around the bounding box
            cv2.rectangle(img, (bboxInfo["bbox"][0], bboxInfo["bbox"][1]),
                          (bboxInfo["bbox"][0] + bboxInfo["bbox"][2], bboxInfo["bbox"][1] + bboxInfo["bbox"][3]),
                          (0, 255, 0), 3)

        # Display the frame with the detected pose and overlay (shirt selection)
        cv2.imshow("Pose Detection - Video", img)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








