import cv2
import numpy as np
import calibration
import os

def get_pink_center(frame):
    """
    HSV色空間を使用してピンク色の物体の中心座標を取得する
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # ピンク色のHSV範囲（HEX: #FF33CC付近）
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # ノイズ除去
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
                return (cx, cy), c, mask
    return None, None, mask

def live_pink_tracking_to_file(K1, dist1, K2, dist2, R, T, camera_height_m=1.557):
    """
    ステレオ視差から高度を計算し、ファイルに書き出す
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    # 投影行列の作成
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))

    capL = cv2.VideoCapture(2)
    capR = cv2.VideoCapture(0)
    
    p_count = 0  # データ数を数えるカウンタ
    p_max_samples = 1000

    # 追記モードで保存
    with open("results_stability/color_+5.txt", "a") as f:
        print(">> 追跡開始。'q'で終了します。")
        
        while p_count < p_max_samples:
            retL, fL = capL.read()
            retR, fR = capR.read()
            if not (retL and retR): break

            resL = get_pink_center(fL)
            resR = get_pink_center(fR)

            if resL[0] is not None and resR[0] is not None:
                cL, cR = resL[0], resR[0]

                # 歪み補正を考慮した座標変換
                pts1 = cv2.undistortPoints(np.array([[float(cL[0])], [float(cL[1])]]), K1, dist1, P=K1)
                pts2 = cv2.undistortPoints(np.array([[float(cR[0])], [float(cR[1])]]), K2, dist2, P=K2)
                
                # 三角測量
                X_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
                X = X_hom[:3] / X_hom[3]
                
                raw_x = X[0][0]
                raw_y = X[1][0]
                raw_z = X[2][0]

                # 【重要】スケールの調整
                # もし raw_z が 50以上なら単位が「cm」の可能性があります。その場合は100で割ります。
                if raw_z > 10: 
                    raw_x /= 100.0
                    raw_y /= 100.0
                    raw_z /= 100.0

                # 高度計算のロジック
                # OpenCVのY軸は下向きなので、カメラより下にある物体は raw_y が正の値になります。
                # 高度 = カメラの地上高 - カメラからの垂直距離(raw_y)
                height_from_ground = camera_height_m - raw_y

                # 表示と保存
                output_line = f"Z(Dist)={raw_z:.3f}m, Y(Rel)={raw_y:.3f}m, Height={height_from_ground:.3f}m"
                print(output_line)
                f.write(output_line + "\n")
                f.flush()
                
                p_count += 1 # データを1つ保存したのでカウント
                print(f"[{p_count}/{p_max_samples}] {output_line}")

                if p_count >= p_max_samples:
                        print(">> 指定したデータ数に達しました。終了します。")
                        break
                    
                # プレビューに描画
                cv2.circle(fL, cL, 7, (0, 0, 255), -1)
                cv2.putText(fL, f"Height: {height_from_ground:.2f}m", (cL[0]-50, cL[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                

            # 映像表示
            cv2.imshow("Left Camera (Tracking)", fL)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # キャリブレーション読み込み
#     try:
#         K1, dist1, K2, dist2, R, T = calibration.load_calibration()
#         live_pink_tracking_to_file(K1, dist1, K2, dist2, R, T, camera_height_m=1.05)
#     except Exception as e:
#         print(f"エラーが発生しました: {e}")