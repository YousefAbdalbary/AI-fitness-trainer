import cv2
import mediapipe as mp
import time
import math
import imutils


class poseDetector():
    start = False
    starttime = 0.0
    lasttime = starttime
    pTime = 0
    elapsed = 0.0
    stretchtime = False
    rep_count = 0

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []  # Initialize lmList as an attribute of the class
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])  # Use self.lmList instead of local lmList
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return img

    def stpwatch(self):
        cTime = time.time()
        elapse = cTime - self.pTime
        self.pTime = cTime

    def findAngle(self, img, p1, p2, p3, draw=True):
        if len(self.lmList) == 0:
            return 0  # Return 0 if lmList is empty

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (2, 2, 2), 3)
            cv2.line(img, (x3, y3), (x2, y2), (2, 2, 2), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)) + "deg", (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            msg = "hold it for 8sec"
            wrn = "Keep your knee bent"
            success = "Successfully completed rep count"

            if ((angle < 120) and (self.start == False) and (self.stretchtime == False)):
                cv2.putText(img, msg, (170, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                self.start = True
                self.starttime = time.time()

            if (self.start):
                self.elapsed = time.time() - self.starttime
                cv2.putText(img, f"timer started :{round(self.elapsed, 0)}", (170, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            if (self.start and angle > 100):
                cv2.putText(img, wrn, (170, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            if (self.start and round(self.elapsed, 0) == 8.0):
                self.start = False
                self.stretchtime = True
                self.rep_count += 1
                cv2.putText(img, success + f"{self.rep_count}", (1, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10, 0), 2)

            if (self.stretchtime and self.start == False):
                cv2.putText(img, success + f"{self.rep_count}", (1, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10, 0), 2)
                cv2.putText(img, "stretch your leg", (5, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10, 0), 2)

            if (self.start == False and angle > 150):
                self.stretchtime = False

        return angle

def main():
    cap = cv2.VideoCapture(0)  # Change to 0 for the default webcam or use another index if you have multiple cameras

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        img = detector.findPose(img, draw=False)
        img = imutils.resize(img, width=720)
        detector.findPosition(img, draw=False)
        detector.findAngle(img, 23, 25, 27, draw=True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.imshow("Live Stream", img)  # Show the live stream window
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
