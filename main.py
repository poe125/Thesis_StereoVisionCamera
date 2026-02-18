# 色マーカー・ArUcoマーカーの高度情報をtxtに書き込む
# 1000個のデータを取得すると終了
import cv2 # type: ignore
import numpy as np # type: ignore
import calibration # type: ignore
import os
import color_marker

def live_stereo_aruco_height(K1, dist1, K2, dist2, R, T, marker_length=0.1, camera_height=1.557):
    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K1 @ P1
    RT = np.hstack((R, T.reshape(3,1)))
    P2 = K2 @ RT

    # カメラの入力設定
    capL = cv2.VideoCapture(2)
    capR = cv2.VideoCapture(0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    count = 0
    max_samples = 1000
    
    print(f">> 記録開始：{max_samples}個のデータを取得します...")

    marker_initial_height = {}  # {id: initial height}
    marker_previous_height = {} # {id: prior height}

    ### 保存先ファイル名 ###
    with open("results_stability/aruco.txt", "a") as f:
        while count < max_samples:
            retL, fL = capL.read()
            retR, fR = capR.read()
            if not (retL and retR):
                print("Failed to get Camera")
                break

            cornersL, idsL, _ = detector.detectMarkers(fL)
            cornersR, idsR, _ = detector.detectMarkers(fR)

            if idsL is not None:
                cv2.aruco.drawDetectedMarkers(fL, cornersL, idsL)
            if idsR is not None:
                cv2.aruco.drawDetectedMarkers(fR, cornersR, idsR)

            # Find matching ID and use triangulation
            if idsL is not None and idsR is not None:
                idsL_list = idsL.flatten().tolist()
                idsR_list = idsR.flatten().tolist()
                common_ids = set(idsL_list).intersection(idsR_list)
                for marker_id in common_ids:
                    idxL = idsL_list.index(marker_id)
                    idxR = idsR_list.index(marker_id)
                    cL = cornersL[idxL][0]
                    cR = cornersR[idxR][0]

                    uL = cL.mean(axis=0)
                    uR = cR.mean(axis=0)

                    pts1 = np.array([[uL[0]],[uL[1]]], dtype=float)
                    pts2 = np.array([[uR[0]],[uR[1]]], dtype=float)

                    X_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
                    X = (X_hom[:3] / X_hom[3]).reshape(3)
                    X_m, Y_m, Z_m = X[0], X[1], X[2]

                    # Text to show on the console
                    text_on_screen = f"ID {marker_id} | X={X_m:.2f} Y={Y_m:.2f} Z={Z_m:.2f} m"
                    # if camera height is known, calculate
                    if camera_height is not None:
                        height_from_ground = camera_height - Y_m

                        if marker_id not in marker_initial_height:
                            marker_initial_height[marker_id] = height_from_ground
                        if marker_id in marker_previous_height:
                            delta_prev = (height_from_ground - marker_previous_height[marker_id]) * 100
                            delta_prev_text = f"({delta_prev:+.1f}cm)"
                        else:
                            delta_prev_text = "(Init)"
                            delta_prev = 0

                        # delta_start = (height_from_ground - marker_initial_height[marker_id]) * 100
                        output_line = f"Z(Dist)={Z_m:.3f}m, Y(Rel)={Y_m:.3f}m, Height={height_from_ground:.3f}m"
                        f.write(output_line + "\n")
                        f.flush()

                        marker_previous_height[marker_id] = height_from_ground
                        
                        count += 1
                        print(f"[{count}/{max_samples}] {output_line}")
                        
                        if count >= max_samples:
                            print(">> 指定したデータ数に達しました。終了します。")
                            break
            both = cv2.hconcat([fL, fR])
            cv2.imshow("Stereo ArUco (L | R)", both)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # camera calibration
    K1, dist1, K2, dist2, R, T = calibration.load_calibration()
    
    # start detecting aruco marker 
    live_stereo_aruco_height(K1, dist1, K2, dist2, R, T, marker_length=0.05, camera_height=1.557)
    
    # start detecting color marker
    color_marker.live_pink_tracking_to_file(K1, dist1, K2, dist2, R, T, camera_height_m=1.557) 
