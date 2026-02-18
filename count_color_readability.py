# 色マーカーの認識率測定をtxtに書き込む

import cv2
import numpy as np
import calibration
import time
import os

def get_pink_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # ピンク色のHSV範囲
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), c
    return None, None

def live_pink_tracking_analysis(K1, dist1, K2, dist2, R, T, camera_height_m=1.05, duration=10):
    os.makedirs("results", exist_ok=True)
    ### 保存先ファイル名 ###
    filepath = "results_readability/color.txt"

    # 投影行列の作成
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))

    capL = cv2.VideoCapture(2)
    capR = cv2.VideoCapture(0)

    total_frames = 0
    success_frames = 0

    print(f">> 10秒間の計測を開始します。保存先: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Time(s), Frame_Index, Success_Flag, Z_Dist, Height\n")
        
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                break

            retL, fL = capL.read()
            retR, fR = capR.read()
            if not (retL and retR): break

            total_frames += 1
            success_flag = 0
            raw_z, height_from_ground = 0.0, 0.0

            resL = get_pink_center(fL)
            resR = get_pink_center(fR)

            # 左右両方でピンクが見つかった場合
            if resL[0] is not None and resR[0] is not None:
                success_frames += 1
                success_flag = 1
                cL, cR = resL[0], resR[0]

                # 三角測量
                pts1 = cv2.undistortPoints(np.array([[float(cL[0])], [float(cL[1])]]), K1, dist1, P=K1)
                pts2 = cv2.undistortPoints(np.array([[float(cR[0])], [float(cR[1])]]), K2, dist2, P=K2)
                X_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
                X = X_hom[:3] / X_hom[3]
                
                raw_y, raw_z = X[1][0], X[2][0]
                if raw_z > 10: # cm単位をmに変換
                    raw_y /= 100.0
                    raw_z /= 100.0
                
                height_from_ground = camera_height_m - raw_y

                # 描画
                cv2.circle(fL, cL, 7, (0, 0, 255), -1)
                cv2.putText(fL, f"Height: {height_from_ground:.2f}m", (cL[0]-50, cL[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ログ保存
            f.write(f"{elapsed_time:.3f}, {total_frames}, {success_flag}, {raw_z:.3f}, {height_from_ground:.3f}\n")

            # 画面表示
            current_rate = (success_frames / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(fL, f"Rate: {current_rate:.1f}% ({success_frames}/{total_frames})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            both = cv2.hconcat([fL, fR])
            cv2.imshow("Pink Tracking Analysis", both)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # 最終統計
        success_rate = (success_frames / total_frames * 100) if total_frames > 0 else 0
        summary = (
            f"\n--- Pink Tracking Statistics ---\n"
            f"Total Frames: {total_frames}\n"
            f"Success Frames: {success_frames}\n"
            f"Success Rate: {success_rate:.2f}%\n"
        )
        f.write(summary)
        print(summary)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        K1, dist1, K2, dist2, R, T = calibration.load_calibration()
        live_pink_tracking_analysis(K1, dist1, K2, dist2, R, T, camera_height_m=1.05, duration=10)
    except Exception as e:
        print(f"エラーが発生しました: {e}")