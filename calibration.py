# ステレオビジョンカメラのキャリブレーション

import cv2 # type: ignore
import numpy as np # type: ignore
import time
import os

### キャリブレーション情報の保存先 ###
CALIB_FILE = "calib_data/calibration.npz"

def live_stereo_calibrate(CHECKERBOARD=(6,9), num_pairs=12, wait_sec=2.0):
    capL = cv2.VideoCapture(2) #change this if "camera not found"
    capR = cv2.VideoCapture(0) #change this if "camera not found"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    last = 0
    print(">> show checkerboard (quit with 'q')")

    while len(objpoints) < num_pairs:
        retL, fL = capL.read()
        retR, fR = capR.read()
        if not (retL and retR):
            raise RuntimeError("No camera found")
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)

        rL, cL = cv2.findChessboardCorners(gL, CHECKERBOARD, None)
        rR, cR = cv2.findChessboardCorners(gR, CHECKERBOARD, None)
        both = cv2.hconcat([fL, fR])
        cv2.imshow("Live Stereo Calib (L | R)", both)

        if rL and rR and (time.time() - last > wait_sec):
            cL2 = cv2.cornerSubPix(gL, cL, (11,11), (-1,-1), criteria)
            cR2 = cv2.cornerSubPix(gR, cR, (11,11), (-1,-1), criteria)
            objpoints.append(objp.copy())
            imgpointsL.append(cL2)
            imgpointsR.append(cR2)
            last = time.time()
            print(f"  captured {len(objpoints)}/{num_pairs}")

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 3:
        raise RuntimeError("lacking available frames")

    print(">> self calibration start...")
    _, K1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, gL.shape[::-1], None, None)
    _, K2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, gR.shape[::-1], None, None)

    print(">> stereo calibration start...")
    flags = cv2.CALIB_FIX_INTRINSIC
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        K1, dist1, K2, dist2, gL.shape[::-1],
        criteria=criteria, flags=flags
    )
    print(">> calibration finished")
    return K1, dist1, K2, dist2, R, T

def save_calibration(K1, dist1, K2, dist2, R, T):
    np.savez(CALIB_FILE, K1=K1, dist1=dist1, K2=K2, dist2=dist2, R=R, T=T)

def load_calibration():
    if os.path.exists(CALIB_FILE):
        data = np.load(CALIB_FILE)
        print("Loaded existing calibration data\n")
        return data["K1"], data["dist1"], data["K2"], data["dist2"], data["R"], data["T"]
    else:
        print("No calibration data. Building new data.\n")
        K1, dist1, K2, dist2, R, T = live_stereo_calibrate(CHECKERBOARD=(9,6), num_pairs=12)
        calib_dir = os.path.dirname(CALIB_FILE)
        if not os.path.exists(calib_dir):
            os.makedirs(calib_dir)
        np.savez(CALIB_FILE, K1=K1, dist1=dist1, K2=K2, dist2=dist2, R=R, T=T)
        print("Saved calibration data.\n")
        return K1, dist1, K2, dist2, R, T
