import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

def main():
    # Use the default webcam (0) as the video source
    cap = cv2.VideoCapture(0)

    # Initialize the PoseDetector class
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    while True:
        # Capture each frame from the webcam
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
            # Get the center of the bounding box
            center = bboxInfo["center"]

            # Draw a circle at the center of the bounding box
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # Implement gesture controls for shirt selection
            # Example: You can track the position of the left/right hand to cycle through shirts.
            # You would use lmList to get hand positions and detect gestures like hands raised above the head.
            pass

        # Display the frame with the detected pose and overlay (e.g., shirt selection)
        cv2.imshow("Pose Detection - Webcam", img)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
