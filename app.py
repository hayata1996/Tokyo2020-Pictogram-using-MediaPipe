import argparse
import traceback
import cv2 as cv
from utils import CvFpsCalc
from utils import logger
from src.video_capture import VideoCapture
from src.pose_estimator import PoseEstimator
from src.drawer import Drawer

logger = logger.get_logger(
    filename=__name__,
    debug=True
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument('--rev_color', action='store_true')
    return parser.parse_args()


def main():
    logger.debug("Start main()")
    args = get_args()
    video_capture = VideoCapture(args.device, args.width, args.height)
    pose_estimator = PoseEstimator(
        args.static_image_mode,
        args.model_complexity,
        args.min_detection_confidence,
        args.min_tracking_confidence
    )
    color = (255, 255, 255) if args.rev_color else (100, 33, 3)
    bg_color = (100, 33, 3) if args.rev_color else (255, 255, 255)
    drawer = Drawer(color, bg_color)
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        try:
            logger.debug("Start while loop")
            display_fps = cvFpsCalc.get()
            frame = video_capture.read()
            if frame is None:
                break

            results = pose_estimator.process(frame)
            if results.pose_landmarks:
                frame = drawer.draw_landmarks(frame, results.pose_landmarks)
                frame = drawer.draw_stick_figure(frame, results.pose_landmarks)

            cv.putText(frame, "FPS:" + str(display_fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow('Output', frame)
            if cv.waitKey(1) == 27:  # ESC key
                break
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())

    video_capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
