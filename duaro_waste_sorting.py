"""
duaro_waste_sorting.py (メインプログラム: ゴミ自動分別・ピッキング制御)

処理の流れ:
1. camera_calibration.json を読み込み (ホモグラフィ行列)
2. duAroに接続・モータON
3. カメラでゴミ箱（白・黄・黒）を自動検出し、ロボット座標を取得
4. カメラでゴミを検出し、カテゴリ（紙/プラ/缶ビン）を判定
5. キー操作 ('s'キー) で自動分別シーケンスを実行:
     [掴む] 物体上空 -> 下降 -> 掴む -> 上昇
     [運ぶ] 対応する箱(白/黒/黄)の上空へ移動
     [捨てる] 箱の中でハンドを開いて投下 -> 上昇 -> 待機位置へ復帰

事前準備:
    - camera.py の Mode 1 でホモグラフィキャリブレーション済みであること
    - duAro側に goto1 プログラムが登録済みであること
    - 白・黄・黒のゴミ箱がカメラの視野内に置いてあること

必要ライブラリ:
    pip install opencv-python numpy ultralytics
"""

import re
import socket
import time

import cv2
import numpy as np

# --- 自作モジュール ---
from camera import (
    load_calibration,
    pixel_to_robot_xy,
    detect_bins_in_frame,
    load_bin_coordinates,
    save_bin_coordinates,
    BIN_COORDINATES_FILE,
)
from duaro_yolo_detector import (
    WasteDetector,
    CATEGORY_PAPER,
    CATEGORY_PLASTIC,
    CATEGORY_CAN_BOTTLE,
)
from air import pump_on, pump_off

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. duAro 接続設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROBOT_IP = "192.168.0.2"
ROBOT_PORT = 23
TIMEOUT = 10
ARM = 1
POSE_NAME = "p1"
PROGRAM_NAME = "goto1"

# ハンド開閉用のASプログラム名 (air.py の pump_on / pump_off を使用)
PUMP_ON_PROGRAM = "pump_on"
PUMP_OFF_PROGRAM = "pump_off"

# カメラ番号 (環境に合わせて変更)
CAMERA_INDEX = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 高さ設定 (単位: mm)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ピック時の高さ（机面からのオフセット）
Z_APPROACH_OFFSET = 60.0  # 物体の上空待機高さ (机 + 60mm)
Z_GRASP_OFFSET = 15.0     # 掴む高さ (机 + 15mm、ワークの厚みに合わせて調整)

# ゴミ箱のZ座標（手動で計測して設定。カメラでは高さ方向を測れないため）
BIN_Z_DROP = 180.0   # 箱の中でゴミを離す高さ (mm)
BIN_Z_SAFE = 280.0   # 箱の上空安全高さ (mm)

# 待機・ホーム位置 (カメラの視野を遮らない安全な位置)
HOME_POS = {"x": 250.0, "y": 0.0, "z": 300.0}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全自動モードの設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTO_NO_DETECT_LIMIT = 5    # 連続未検出回数がこの値に達したら「全完了」と判断
AUTO_STABILIZE_WAIT = 1.0   # 分別後、次の検出前の待機時間（秒）
AUTO_MAX_FAILURES = 3       # 連続失敗がこの値に達したら自動停止


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 分別マッピング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY_TO_BIN = {
    CATEGORY_PAPER:      "白の箱",   # 紙ごみ   -> 白の箱
    CATEGORY_PLASTIC:    "黄の箱",   # プラスチック -> 黄の箱
    CATEGORY_CAN_BOTTLE: "黒の箱",   # 缶ビン   -> 黒の箱
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. duAro 通信・低レベル制御
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def connect(ip, port, timeout):
    """duAroにTelnet接続し、自動ログインする"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((ip, port))
    time.sleep(0.5)
    resp = s.recv(2048).decode("shift_jis", errors="ignore")
    if "login:" in resp.lower() or "as" in resp.lower():
        s.sendall(b"as\r\n")
        time.sleep(0.5)
        resp = s.recv(2048).decode("shift_jis", errors="ignore")
        if "password:" in resp.lower():
            s.sendall(b"\r\n")
            time.sleep(0.5)
            s.recv(2048)
    return s


def disconnect(sock):
    try:
        sock.close()
    except Exception:
        pass


def flush_buffer(sock):
    """受信残余バッファ（ゴミデータ）をクリア"""
    sock.settimeout(0.001)
    try:
        while sock.recv(1024):
            pass
    except Exception:
        pass


def send_cmd(sock, command, wait=0.3):
    """コマンド送信 → 応答受信"""
    try:
        flush_buffer(sock)
        sock.settimeout(2.0)
        sock.sendall((command + "\r\n").encode("shift_jis"))
        time.sleep(wait)
        sock.settimeout(0.5)
        resp = ""
        try:
            while True:
                chunk = sock.recv(4096).decode("shift_jis", errors="ignore")
                if not chunk:
                    break
                resp += chunk
        except socket.timeout:
            pass
        return resp
    except Exception as e:
        print(f"[ERROR] 送信/受信エラー: {e}")
        return ""


_ERROR_PATTERN = re.compile(r"(\([EP]\d{4}\)|ホールド|途中停止|動作範囲外|ABORT)")


def wait_for_completion(sock, max_wait=30):
    """
    duAroが完了メッセージを返すまで待つ。
    エラーを検知した場合は即座に False を返す。
    """
    start = time.time()
    full_resp = ""
    sock.settimeout(0.5)
    while time.time() - start < max_wait:
        try:
            chunk = sock.recv(2048).decode("shift_jis", errors="ignore")
            if not chunk:
                break
            full_resp += chunk
            if "プログラムが終了しました" in full_resp:
                print(f"[INFO] 移動完了")
                return True
            if _ERROR_PATTERN.search(full_resp):
                print(f"[ERROR] ロボット異常停止:\n{full_resp.strip()}")
                return False
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[ERROR] 受信エラー: {e}")
            return False
    print(f"[WARN] {max_wait}秒以内に完了メッセージを検知できませんでした")
    return False


def motor_on(sock, arm=ARM):
    """モータ電源をONにする"""
    print(f"[INFO] モータ電源ON: ZPOWER {arm}: ON")
    resp = send_cmd(sock, f"ZPOWER {arm}: ON", wait=1.0)
    print(f"[RECV] {resp.strip()}")
    return resp


def set_point(sock, x, y, z, o=0.0, a=0.0, t=0.0):
    """
    POINT指令 + TRANS関数で座標を設定する。
    「変更？」プロンプトに空Enter で確定。
    """
    cmd = (
        f"POINT {ARM}: {POSE_NAME} = "
        f"TRANS({x:.2f},{y:.2f},{z:.2f},{o:.2f},{a:.2f},{t:.2f})"
    )
    resp1 = send_cmd(sock, cmd, wait=0.5)

    # 「変更？」が届くまで最大3秒待つ
    deadline = time.time() + 3.0
    while "変更" not in resp1 and time.time() < deadline:
        time.sleep(0.2)
        sock.settimeout(0.2)
        try:
            chunk = sock.recv(4096).decode("shift_jis", errors="ignore")
            resp1 += chunk
        except socket.timeout:
            pass
        except Exception:
            break

    # 空Enter で確定
    send_cmd(sock, "", wait=0.3)
    return resp1


def move_to_xyz(sock, x, y, z):
    """指定座標 (x, y, z) へ移動して完了を待つ"""
    print(f"[INFO] 移動: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
    set_point(sock, x, y, z)

    # EXECUTE でプログラム実行
    try:
        flush_buffer(sock)
        sock.settimeout(2.0)
        sock.sendall(f"EXECUTE {ARM}:{PROGRAM_NAME}\r\n".encode("shift_jis"))
    except Exception as e:
        print(f"[ERROR] EXECUTE 送信エラー: {e}")
        return False

    return wait_for_completion(sock)


def gripper_control(sock, grasp=True):
    """ハンドの開閉（空気ポンプ: pump_on / pump_off）"""
    if grasp:
        print("[INFO] ハンド閉じる(掴む): pump_on")
        success, _ = pump_on(sock, arm=ARM, program_name=PUMP_ON_PROGRAM)
    else:
        print("[INFO] ハンド開く(離す): pump_off")
        success, _ = pump_off(sock, arm=ARM, program_name=PUMP_OFF_PROGRAM)
    if not success:
        print("[WARN] ハンド開閉コマンドの完了確認に失敗しました")
    return success


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ゴミ箱自動検出（起動時に実行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def auto_detect_bins(cap, h_matrix):
    """
    カメラ映像から白・黄・黒のゴミ箱を色で自動検出し、
    ロボット座標 (X, Y) を返す。

    複数フレームの検出結果を平均化してノイズを低減する。
    """
    print("\n[INFO] ゴミ箱を自動検出しています...")
    print("[INFO] 白・黄・黒のゴミ箱がカメラに映っていることを確認してください")

    # カメラの露出が安定するまで読み飛ばす
    for _ in range(30):
        cap.read()

    # 複数フレームで検出し、中央値を取る（安定化）
    NUM_SAMPLES = 10
    samples = {name: [] for name in ["白の箱", "黄の箱", "黒の箱"]}

    for i in range(NUM_SAMPLES):
        ret, frame = cap.read()
        if not ret:
            continue
        centers, _ = detect_bins_in_frame(frame)
        for bin_name, center in centers.items():
            if center is not None:
                samples[bin_name].append(center)
        time.sleep(0.05)

    # 各箱の中央値を計算 → ロボット座標に変換
    bin_coords = {}
    for bin_name, pts in samples.items():
        if len(pts) < NUM_SAMPLES // 2:
            # 半数以上のフレームで検出できなかった場合はスキップ
            print(f"[WARN] {bin_name} が安定して検出できませんでした "
                  f"({len(pts)}/{NUM_SAMPLES} フレームで検出)")
            continue

        # ピクセル座標の中央値
        px_median = int(np.median([p[0] for p in pts]))
        py_median = int(np.median([p[1] for p in pts]))

        # ロボット座標に変換
        rx, ry = pixel_to_robot_xy(px_median, py_median, h_matrix)
        bin_coords[bin_name] = {
            "x": round(rx, 2),
            "y": round(ry, 2),
            "z_drop": BIN_Z_DROP,
            "z_safe": BIN_Z_SAFE,
        }
        print(f"[INFO] {bin_name}: X={rx:.1f}, Y={ry:.1f} mm "
              f"(pixel: {px_median}, {py_median})")

    return bin_coords


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 分別ピッキングシーケンス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def execute_sorting(sock, target_info, bin_coords, h_matrix, table_z):
    """
    検出された物体を掴み、対応する箱へ運んで捨てる。

    sock        : duAroソケット
    target_info : WasteDetector.detect() の戻り値
    bin_coords  : {"白の箱": {"x":..., "y":..., "z_drop":..., "z_safe":...}, ...}
    h_matrix    : ホモグラフィ行列
    table_z     : 机のZ座標 (mm)
    """
    category = target_info["category"]
    box_name = CATEGORY_TO_BIN.get(category)

    if not box_name:
        print(f"[WARN] カテゴリ '{category}' に対応する箱が定義されていません")
        return False

    if box_name not in bin_coords:
        print(f"[WARN] {box_name} の座標が取得できていません（未検出）")
        return False

    bin_pos = bin_coords[box_name]

    # 物体のピクセル座標 → ロボット座標
    px, py = target_info["pixel"]
    rx, ry = pixel_to_robot_xy(px, py, h_matrix)
    z_approach = table_z + Z_APPROACH_OFFSET
    z_grasp = table_z + Z_GRASP_OFFSET

    print("\n" + "=" * 50)
    print(f"【分別開始】 対象: {target_info['class']} ({category})")
    print(f"  ・物体位置   : X={rx:.1f}, Y={ry:.1f}, Z={z_grasp:.1f} mm")
    print(f"  ・搬送先     : {box_name} (X={bin_pos['x']:.1f}, Y={bin_pos['y']:.1f})")
    print("=" * 50)

    # --- 1. ピック動作（掴む） ---
    # ① まずハンドを開く
    gripper_control(sock, grasp=False)

    # ② 物体の上空へ移動（衝突防止）
    print(">> [1/7] 物体上空へ移動")
    if not move_to_xyz(sock, rx, ry, z_approach):
        return False

    # ③ 掴む高さまで下降
    print(">> [2/7] 下降")
    if not move_to_xyz(sock, rx, ry, z_grasp):
        return False

    # ④ チャックを閉じて掴む
    print(">> [3/7] 掴む")
    gripper_control(sock, grasp=True)

    # ⑤ 上空へ持ち上げる
    print(">> [4/7] 持ち上げ")
    if not move_to_xyz(sock, rx, ry, z_approach):
        return False

    # --- 2. プレイス動作（箱へ捨てる） ---
    # ⑥ 箱の上空安全高さへ運ぶ
    print(f">> [5/7] {box_name}の上空へ移動")
    if not move_to_xyz(sock, bin_pos["x"], bin_pos["y"], bin_pos["z_safe"]):
        return False

    # ⑦ 箱の投下高さまで下げる
    print(f">> [6/7] 投下高さへ下降")
    if not move_to_xyz(sock, bin_pos["x"], bin_pos["y"], bin_pos["z_drop"]):
        return False

    # ⑧ ハンドを開いてゴミを落とす
    print(f">> [7/7] 投下（{box_name}に捨てる）")
    gripper_control(sock, grasp=False)

    # ⑨ 箱の上空安全高さへ退避
    move_to_xyz(sock, bin_pos["x"], bin_pos["y"], bin_pos["z_safe"])

    # ⑩ 待機位置へ戻る
    print(">> 待機位置へ復帰")
    move_to_xyz(sock, HOME_POS["x"], HOME_POS["y"], HOME_POS["z"])

    print("=== 分別完了 ===\n")
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.5. 全自動分別ループ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def auto_sorting_loop(sock, cap, detector, bin_coords, h_matrix, table_z, window_name):
    """
    全自動分別ループ。
    ゴミを検出→分別→次のゴミを検出→分別… をゴミがなくなるまで繰り返す。
    'q' キーで緊急停止可能。

    戻り値: 分別完了した個数
    """
    sorted_count = 0
    no_detect_count = 0
    fail_count = 0

    print("\n" + "=" * 50)
    print("【全自動分別モード開始】")
    print("  ゴミがなくなるまで自動で分別を繰り返します")
    print("  'q' キーで緊急停止")
    print("=" * 50 + "\n")

    while no_detect_count < AUTO_NO_DETECT_LIMIT:
        # フレーム取得
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] カメラからフレームを取得できませんでした")
            break

        # YOLO で物体検出
        target, display_frame = detector.detect(frame)

        # 画面にステータス表示
        status = f"AUTO MODE | Sorted: {sorted_count} | Scanning: {no_detect_count}/{AUTO_NO_DETECT_LIMIT}"
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if target:
            cat = target["category"]
            box_name = CATEGORY_TO_BIN.get(cat, "未設定")
            cv2.putText(display_frame, f"Next: {cat} -> [{box_name}]", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Scanning for waste...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        cv2.imshow(window_name, display_frame)

        # 'q' キーで緊急停止
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] 全自動モードを緊急停止しました")
            break

        # --- 未検出の場合 ---
        if target is None:
            no_detect_count += 1
            time.sleep(0.3)
            continue

        # --- 検出あり → 未検出カウンタをリセット ---
        no_detect_count = 0

        # 分別先の箱が検出済みか確認
        cat = target["category"]
        needed_bin = CATEGORY_TO_BIN.get(cat)
        if needed_bin and needed_bin not in bin_coords:
            print(f"[ERROR] {needed_bin} が未検出のため自動モードを停止します")
            break

        # 分別実行
        success = execute_sorting(sock, target, bin_coords, h_matrix, table_z)

        if success:
            sorted_count += 1
            fail_count = 0
            print(f"[INFO] 【{sorted_count}個目 完了】\n")
        else:
            fail_count += 1
            print(f"[WARN] 分別失敗 ({fail_count}/{AUTO_MAX_FAILURES})")
            if fail_count >= AUTO_MAX_FAILURES:
                print("[ERROR] 連続失敗の上限に達しました。自動モードを停止します")
                break

        # 分別後、アームが退避してカメラが安定するまで待機
        time.sleep(AUTO_STABILIZE_WAIT)

    else:
        # while条件で正常終了 = 連続未検出で全完了
        print(f"\n[INFO] {AUTO_NO_DETECT_LIMIT}回連続で未検出 → ゴミがなくなりました")

    print("\n" + "=" * 50)
    print(f"【全自動分別完了】 合計 {sorted_count} 個のゴミを分別しました")
    print("=" * 50 + "\n")

    return sorted_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. メインループ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # --- A. キャリブレーション読み込み ---
    try:
        H_matrix, table_z = load_calibration()
        print(f"[INFO] キャリブレーション読み込み完了 (机Z={table_z:.1f} mm)")
    except FileNotFoundError as e:
        print(e)
        print("[INFO] 先に camera.py の Mode 1 を実行してください")
        raise SystemExit(1)

    # --- B. カメラ起動 ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] カメラ(index={CAMERA_INDEX})を開けませんでした")
        raise SystemExit(1)

    # --- C. ゴミ箱を自動検出 ---
    bin_coords = auto_detect_bins(cap, H_matrix)

    if not bin_coords:
        print("[ERROR] ゴミ箱が1つも検出できませんでした")
        print("[INFO] カメラに白・黄・黒の箱が映るように配置してから再実行してください")
        cap.release()
        raise SystemExit(1)

    missing_bins = [name for name in CATEGORY_TO_BIN.values() if name not in bin_coords]
    if missing_bins:
        print(f"[WARN] 以下の箱が検出できていません: {', '.join(missing_bins)}")
        print("[WARN] 該当カテゴリのゴミは分別できません")

    # 検出結果を保存（次回の参照用）
    save_bin_coordinates(bin_coords)

    # --- D. duAro接続 ---
    print(f"\n[INFO] duAro ({ROBOT_IP}:{ROBOT_PORT}) に接続中...")
    sock = None
    try:
        sock = connect(ROBOT_IP, ROBOT_PORT, TIMEOUT)
        print("[INFO] 接続成功")
    except ConnectionRefusedError:
        print(f"[ERROR] 接続拒否: {ROBOT_IP}:{ROBOT_PORT} に接続できません")
        cap.release()
        raise SystemExit(1)
    except socket.timeout:
        print(f"[ERROR] タイムアウト: {TIMEOUT}秒以内に応答がありません")
        cap.release()
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] 接続エラー: {e}")
        cap.release()
        raise SystemExit(1)

    # --- E. モータON ---
    motor_on(sock, arm=ARM)

    # --- F. YOLO検出器の初期化 ---
    detector = WasteDetector()

    # --- G. メインループ ---
    window_name = "duAro Waste Sorting [s: Sort, a: Auto, q: Quit]"
    cv2.namedWindow(window_name)

    print("\n" + "=" * 50)
    print("【duAro ゴミ自動分別システム】")
    print("=" * 50)
    print(f"  検出済みゴミ箱: {', '.join(bin_coords.keys())}")
    print()
    print("  [操作方法]")
    print("  ・カメラの視野にゴミを置く")
    print("  ・'s' キー → 1個だけ分別（手動モード）")
    print("  ・'a' キー → 全自動分別（ゴミがなくなるまで繰り返す）")
    print("  ・'q' キー → 終了")
    print("=" * 50 + "\n")

    # duAro側の前提プログラム確認メッセージ
    print(f"[INFO] duAro側に以下のASプログラムが登録されている必要があります:")
    print(f"       1) .PROGRAM {PROGRAM_NAME}()  / JMOVE {POSE_NAME}  / .END")
    print(f"       2) .PROGRAM {PUMP_ON_PROGRAM}()   / OPENI 1   / .END")
    print(f"       3) .PROGRAM {PUMP_OFF_PROGRAM}()  / CLOSEI 1  / .END\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] カメラからフレームを取得できませんでした")
                break

            # YOLO で物体検出・カテゴリ判定
            target, display_frame = detector.detect(frame)

            # 画面上に案内を表示
            if target:
                cat = target["category"]
                box_name = CATEGORY_TO_BIN.get(cat, "未設定")
                conf = target["conf"]
                px, py = target["pixel"]

                # ロボット座標も画面に表示
                rx, ry = pixel_to_robot_xy(px, py, H_matrix)
                guide = (
                    f"TARGET: {cat} -> [{box_name}] "
                    f"(conf:{conf:.2f}, X={rx:.0f}, Y={ry:.0f})"
                )
                cv2.putText(display_frame, guide, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "'s': Sort 1 / 'a': Auto-sort all", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No target detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                if target is None:
                    print("[WARN] 分別対象の物体が検出されていません")
                    continue

                # 分別先の箱が検出済みか確認
                cat = target["category"]
                needed_bin = CATEGORY_TO_BIN.get(cat)
                if needed_bin and needed_bin not in bin_coords:
                    print(f"[WARN] {needed_bin} が検出されていないため分別できません")
                    continue

                execute_sorting(sock, target, bin_coords, H_matrix, table_z)

            elif key == ord("a"):
                auto_sorting_loop(sock, cap, detector, bin_coords, H_matrix, table_z, window_name)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C で中断されました")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sock:
            disconnect(sock)
        print("[INFO] ロボットから切断しました")
