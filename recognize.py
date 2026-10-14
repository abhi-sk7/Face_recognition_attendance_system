import time

import cv2

from db import init_db
from engine import Engine


def main():
    init_db()
    engine = Engine()
    engine.start()
    try:
        while True:
            with engine.data_lock:
                frame = None if engine.annot_frame is None else engine.annot_frame.copy()
            if frame is None:
                time.sleep(0.03)
                continue
            cv2.imshow("Face Attendance (ESC = exit)", frame)
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
