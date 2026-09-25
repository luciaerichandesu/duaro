"""
camera.py — カメラキャリブレーション & ゴミ箱自動検出モジュール

Mode 1: ホモグラフィキャリブレーション
    Webカメラの映像上で机の四隅をクリックし、それぞれに対応する
    duAro側の実座標(X, Y, mm)を入力することで、
    「カメラのピクセル座標 → duAroのXY座標」への変換行列(ホモグラフィ)
    を計算して camera_calibration.json に保存する。

Mode 2: ゴミ箱自動検出
    キャリブレーション済みのホモグラフィ行列を使い、
    カメラ映像から白・黄・黒のゴミ箱を色で自動検出して
    ロボット座標に変換し bin_coordinates.json に保存する。

事前準備 (Mode 1):
    - duAroを実際に机の四隅（角に置いた物体の位置など）へジョグ/移動させ、
      その都度 krtermで `HERE p_now` を実行して X, Y の値をメモしておく。
    - 4隅の順番を、カメラでクリックする順番と一致させること。
      本スクリプトでは「左上→右上→右下→左下」の順でクリックする想定。

必要ライブラリ:
    pip install opencv-python numpy
"""

import json
import os

import cv2
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALIBRATION_FILE = "camera_calibration.json"
BIN_COORDINATES_FILE = "bin_coordinates.json"
CORNER_LABELS = ["左上", "右上", "右下", "左下"]

# ゴミ箱の色検出パラメータ (HSV)
# ※ 照明環境に応じて lower / upper を調整してください
BIN_COLOR_RANGES = {
    "白の箱": {
        "lower": np.array([0, 0, 180]),
        "upper": np.array([180, 60, 255]),
        "display_bgr": (255, 255, 255),
    },
    "黄の箱": {
        "lower": np.array([18, 70, 70]),
        "upper": np.array([40, 255, 255]),
        "display_bgr": (0, 255, 255),
    },
    "黒の箱": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 120, 50]),
        "display_bgr": (80, 80, 80),
    },
}

# 箱として認識する最小面積 (ピクセル)
MIN_BIN_AREA = 3000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ホモグラフィキャリブレーション（既存機能）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pick_corners_from_camera(camera_index=1):
    """
    Webカメラを開き、映像上で机の四隅を順にクリックさせる。
    戻り値: [(px1,py1), (px2,py2), (px3,py3), (px4,py4)]
    """
    clicked_points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"[INFO] {CORNER_LABELS[len(clicked_points) - 1]} をクリック: ({x}, {y})")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ(index={camera_index})を開けませんでした")

    window_name = "duAro camera calibration - click 4 corners (q: quit, r: reset)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("[INFO] 映像上で机の四隅を「左上→右上→右下→左下」の順にクリックしてください")
    print("[INFO] やり直したい場合は 'r' キー、途中でやめる場合は 'q' キー")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("カメラからフレームを取得できませんでした")

            # クリック済みの点と番号を描画
            for i, (px, py) in enumerate(clicked_points):
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"{i + 1}:{CORNER_LABELS[i]}", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                clicked_points.clear()
                print("[INFO] クリックをリセットしました")
            elif key == ord("q"):
                raise KeyboardInterrupt("ユーザーによる中断")

            if len(clicked_points) == 4:
                print("[INFO] 4点クリックされました。何かキーを押すと確定します（'r'でやり直し）")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord("r"):
                    clicked_points.clear()
                    print("[INFO] クリックをリセットしました")
                    continue
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return clicked_points


def input_robot_xy(label):
    """コンソールからロボット側のX, Y座標(mm)を入力させる"""
    while True:
        raw = input(f"  {label} に対応するduAro座標を入力 (X Y / mm mm, 空白区切り): ").strip()
        parts = raw.split()
        if len(parts) != 2:
            print("  [WARN] X Y の2値で入力してください")
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            return x, y
        except ValueError:
            print("  [WARN] 数値として解釈できません。もう一度入力してください")


def compute_homography(pixel_points, robot_points):
    """
    4組の対応点から、ピクセル座標→ロボットXY座標へのホモグラフィ行列を計算する。
    """
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(robot_points, dtype=np.float32)
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None:
        raise RuntimeError("ホモグラフィ行列の計算に失敗しました")
    return matrix


def save_calibration(matrix, table_z_mm, path=CALIBRATION_FILE):
    data = {
        "homography": matrix.tolist(),
        "table_z_mm": table_z_mm,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] キャリブレーション結果を {path} に保存しました")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ユーティリティ（他ファイルからも import して使う）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_calibration(path=CALIBRATION_FILE):
    """camera_calibration.json からホモグラフィ行列と机のZ座標を読み込む"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] '{path}' が見つかりません。先に Mode 1 でキャリブレーションを実行してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = np.array(data["homography"], dtype=np.float64)
    table_z = float(data["table_z_mm"])
    return matrix, table_z


def pixel_to_robot_xy(px, py, homography_matrix):
    """ピクセル座標 (px, py) をロボット座標 (X, Y) に変換する"""
    vec = np.array([px, py, 1.0], dtype=np.float64)
    r_vec = np.dot(homography_matrix, vec)
    return float(r_vec[0] / r_vec[2]), float(r_vec[1] / r_vec[2])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ゴミ箱の色検出（Mode 2）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_bins_in_frame(frame):
    """
    1フレームから白・黄・黒の箱を HSV 色検出で見つけ、
    各箱の中心ピクセル座標を返す。

    戻り値:
        dict: {"白の箱": (cx, cy), "黄の箱": (cx, cy), "黒の箱": (cx, cy)}
              検出できなかった箱の値は None
        dict: {"白の箱": contour, ...} — 描画用の輪郭
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    kernel = np.ones((7, 7), np.uint8)

    centers = {}
    contours_map = {}

    for bin_name, cfg in BIN_COLOR_RANGES.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_BIN_AREA and area > best_area:
                best_contour = cnt
                best_area = area

        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers[bin_name] = (cx, cy)
                contours_map[bin_name] = best_contour
            else:
                centers[bin_name] = None
                contours_map[bin_name] = None
        else:
            centers[bin_name] = None
            contours_map[bin_name] = None

    return centers, contours_map


def detect_bins_from_camera(camera_index=1, homography_matrix=None):
    """
    カメラ映像からゴミ箱を色で検出し、ロボット座標に変換して返す。
    対話的に確認・やり直しが可能。

    戻り値:
        dict: {"白の箱": {"x": ..., "y": ...}, ...}
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ(index={camera_index})を開けませんでした")

    # カメラの自動露出が安定するまで数フレーム読み飛ばす
    for _ in range(30):
        cap.read()

    window_name = "Bin Detection (s: save, r: retry, q: quit)"
    cv2.namedWindow(window_name)

    print("\n[INFO] ゴミ箱の自動検出中...")
    print("[INFO] 白・黄・黒の箱がカメラに映るように配置してください")
    print("[INFO] 's' で確定保存 / 'r' で再検出 / 'q' で中断")

    detected_bins = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("カメラからフレームを取得できませんでした")

            centers, contours_map = detect_bins_in_frame(frame)
            display = frame.copy()

            # 検出結果を描画
            status_y = 30
            all_found = True
            for bin_name, cfg in BIN_COLOR_RANGES.items():
                color = cfg["display_bgr"]
                center = centers.get(bin_name)
                contour = contours_map.get(bin_name)

                if center is not None and contour is not None:
                    cv2.drawContours(display, [contour], -1, color, 2)
                    cv2.circle(display, center, 8, color, -1)

                    # ロボット座標に変換して表示
                    if homography_matrix is not None:
                        rx, ry = pixel_to_robot_xy(center[0], center[1], homography_matrix)
                        label = f"{bin_name}: px({center[0]},{center[1]}) -> robot({rx:.1f},{ry:.1f})"
                    else:
                        label = f"{bin_name}: px({center[0]},{center[1]})"

                    cv2.putText(display, f"OK {label}", (10, status_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.putText(display, f"NG {bin_name}: 未検出", (10, status_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    all_found = False

                status_y += 25

            # 全箱検出済みなら緑で案内表示
            if all_found:
                cv2.putText(display, "ALL BINS DETECTED - Press 's' to save",
                            (10, status_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                raise KeyboardInterrupt("ユーザーによる中断")

            elif key == ord("s"):
                # 現在の検出結果を確定
                detected_bins = {}
                missing = []
                for bin_name in BIN_COLOR_RANGES:
                    center = centers.get(bin_name)
                    if center is not None and homography_matrix is not None:
                        rx, ry = pixel_to_robot_xy(center[0], center[1], homography_matrix)
                        detected_bins[bin_name] = {"x": round(rx, 2), "y": round(ry, 2)}
                    elif center is not None:
                        detected_bins[bin_name] = {"px": center[0], "py": center[1]}
                    else:
                        missing.append(bin_name)

                if missing:
                    print(f"[WARN] 以下の箱が検出できていません: {', '.join(missing)}")
                    ans = input("  検出できた箱だけで続行しますか？ (y/n): ").strip().lower()
                    if ans != "y":
                        print("[INFO] 再検出します。箱を映し直してください")
                        continue
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return detected_bins


def save_bin_coordinates(bin_coords, path=BIN_COORDINATES_FILE):
    """ゴミ箱のロボット座標を JSON に保存する"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bin_coords, f, ensure_ascii=False, indent=2)
    print(f"[INFO] ゴミ箱座標を {path} に保存しました")


def load_bin_coordinates(path=BIN_COORDINATES_FILE):
    """ゴミ箱のロボット座標を JSON から読み込む"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] '{path}' が見つかりません。先にゴミ箱検出を実行してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. メイン（対話メニュー）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_homography_calibration(camera_index):
    """Mode 1: ホモグラフィキャリブレーション"""
    pixel_points = pick_corners_from_camera(camera_index)
    print(f"[INFO] クリックされたピクセル座標: {pixel_points}")

    print("\n続いて、各隅に対応する duAro 側の実座標(X, Y, mm)を入力してください。")
    print("（あらかじめduAroをその角の位置へ動かし、krtermの `HERE` で確認しておいた値を入力）")

    robot_points = []
    for label in CORNER_LABELS:
        x, y = input_robot_xy(label)
        robot_points.append((x, y))

    table_z_raw = input("\n机の高さ(duAro座標系でのZ値, mm)を入力: ").strip()
    table_z_mm = float(table_z_raw) if table_z_raw else 0.0

    matrix = compute_homography(pixel_points, robot_points)
    print(f"[INFO] 計算されたホモグラフィ行列:\n{matrix}")

    save_calibration(matrix, table_z_mm)
    return matrix, table_z_mm


def run_bin_detection(camera_index):
    """Mode 2: ゴミ箱自動検出"""
    h_matrix, _ = load_calibration()
    bin_coords = detect_bins_from_camera(camera_index, homography_matrix=h_matrix)

    if bin_coords:
        print("\n[INFO] 検出されたゴミ箱座標:")
        for name, coord in bin_coords.items():
            print(f"  {name}: X={coord['x']:.2f}, Y={coord['y']:.2f}")
        save_bin_coordinates(bin_coords)
    else:
        print("[WARN] ゴミ箱が1つも検出されませんでした")


if __name__ == "__main__":
    print("=" * 50)
    print("  duAro カメラセットアップ")
    print("=" * 50)

    camera_index = input("カメラ番号を入力（分からなければ 1 のままEnter）: ").strip()
    camera_index = int(camera_index) if camera_index else 1

    print("\n  1. ホモグラフィキャリブレーション（机の四隅）")
    print("  2. ゴミ箱の自動検出（白・黄・黒の箱）")
    print("  3. 両方実行（1 → 2 の順）")
    print("  q. 終了")
    choice = input("\n選択: ").strip()

    if choice == "1":
        run_homography_calibration(camera_index)
    elif choice == "2":
        run_bin_detection(camera_index)
    elif choice == "3":
        run_homography_calibration(camera_index)
        print("\n--- 続いてゴミ箱の検出に進みます ---\n")
        run_bin_detection(camera_index)
    elif choice == "q":
        print("[INFO] 終了")
    else:
        print("[ERROR] 無効な選択です")

    print("\n[INFO] カメラセットアップ完了")