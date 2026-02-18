# ArUcoマーカーの認識率測定をtxtに書きこむ

import cv2 # type: ignore
import numpy as np # type: ignore
import calibration # type: ignore
import time
import os

def live_stereo_aruco_height(K1, dist1, K2, dist2, R, T, marker_length=0.1, camera_height=1.557, duration=10):
    # --- 前処理 (投影行列の設定など) ---
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K1 @ P1
    RT = np.hstack((R, T.reshape(3,1)))
    P2 = K2 @ RT

    capL = cv2.VideoCapture(2)
    capR = cv2.VideoCapture(0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # 結果保存用の設定
    os.makedirs("results", exist_ok=True)
    filepath = "results_readability/aruco_test.txt"
    
    total_frames = 0
    success_frames = 0
    
    print(f">> Recording for {duration}s. Saving to {filepath}...")

    with open(filepath, "w", encoding="utf-8") as f:
        # ヘッダーの書き込み
        f.write("Time(s), Frame_Index, Success_Flag, Detected_IDs\n")
        
        start_time = time.time()

        while True:
            current_time = time.time() - start_time
            if current_time > duration:
                break

            retL, fL = capL.read()
            retR, fR = capR.read()
            if not (retL and retR):
                break

            total_frames += 1
            success_flag = 0 # 0: 失敗, 1: 成功
            common_ids_found = []

            cornersL, idsL, _ = detector.detectMarkers(fL)
            cornersR, idsR, _ = detector.detectMarkers(fR)

            if idsL is not None and idsR is not None:
                idsL_list = idsL.flatten().tolist()
                idsR_list = idsR.flatten().tolist()
                common_ids = set(idsL_list).intersection(idsR_list)
                
                if common_ids:
                    success_frames += 1
                    success_flag = 1
                    common_ids_found = list(common_ids)
                    
                    # (三角測量などの計算が必要な場合はここに記述)

            # --- テキストファイルへの書き出し ---
            # フォーマット: 経過時間, フレーム番号, 成功フラグ, 見つかったIDリスト
            f.write(f"{current_time:.3f}, {total_frames}, {success_flag}, \"{common_ids_found}\"\n")

            # プレビュー表示
            if idsL is not None: cv2.aruco.drawDetectedMarkers(fL, cornersL, idsL)
            cv2.putText(fL, f"Frames: {total_frames} | Success: {success_frames}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            both = cv2.hconcat([fL, fR])
            cv2.imshow("Stereo Aruco Recording", both)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- 最後に統計を追記 ---
        success_rate = (success_frames / total_frames * 100) if total_frames > 0 else 0
        summary = (
            f"\n--- Final Statistics ---\n"
            f"Duration: {duration}s\n"
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
    K1, dist1, K2, dist2, R, T = calibration.load_calibration()
    live_stereo_aruco_height(K1, dist1, K2, dist2, R, T, marker_length=0.12, camera_height=1.05, duration=10)