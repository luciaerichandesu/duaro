"""
duaro_waste_sorting.py (メインプログラム: ゴミ自動分別・ピッキング制御 / 双腕対応)

処理の流れ:
1. camera_calibration.json を読み込み (アームごとのホモグラフィ行列・左右の境界)
2. duAroに接続・モータON (アーム1・アーム2 両方)
3. カメラでゴミ箱（赤・黄・青）を自動検出し、各アームから見たロボット座標を取得
4. カメラでゴミを検出し、カテゴリ（紙/プラ/缶ビン）を判定
5. 物体の位置でアームを選ぶ:
     画面の右側 -> アーム1 (下に付いている右腕)
     画面の左側 -> アーム2 (上に付いている左腕)
6. キー操作 ('s'キー) で自動分別シーケンスを実行:
     [掴む] 物体上空 -> 下降 -> 掴む -> 上昇
     [運ぶ] 対応する箱(赤/黄/青)の上空へ移動 (箱は中央に置く前提)
     [捨てる] 箱の中でハンドを開いて投下 -> 上昇 -> 待機位置へ復帰
   ※ 衝突防止のため、動かすアームは常に1本だけ。もう片方は待機位置で待つ。

事前準備:
    - camera.py の Mode 1 で双腕6点キャリブレーション済みであること
    - duAro側に goto1 / pump_on / pump_off プログラムが登録済みであること
    - 赤・黄・青のゴミ箱がカメラの視野内 (中央) に置いてあること

必要ライブラリ:
    pip install opencv-python numpy ultralytics
"""

import re
import socket
import time

try:
    import msvcrt
    _MSVCRT_AVAILABLE = True
except ImportError:
    _MSVCRT_AVAILABLE = False

import cv2
import numpy as np


def get_pressed_key(wait_ms=20):
    """
    OpenCVウィンドウのキー入力とターミナル（コンソール）のキー入力を両方受け付ける。
    全角文字（'ｓ','ａ','ｑ'）や半角文字（'s','a','q'）、大文字・ESCキーも統一判定。
    """
    key = cv2.waitKey(wait_ms) & 0xFF

    # 1. ターミナル側でキーが押されている場合 (msvcrt)
    if _MSVCRT_AVAILABLE and msvcrt.kbhit():
        try:
            char = msvcrt.getwch().lower()
            if char in ("s", "ｓ"):
                return ord("s")
            elif char in ("a", "ａ"):
                return ord("a")
            elif char in ("q", "ｑ", "\x1b"):
                return ord("q")
        except Exception:
            pass

    # 2. OpenCV ウィンドウ側でのキー判定
    if key in (ord("s"), ord("S")):
        return ord("s")
    elif key in (ord("a"), ord("A")):
        return ord("a")
    elif key in (ord("q"), ord("Q"), 27):
        return ord("q")

    return key

# --- 自作モジュール ---
from camera import (
    load_calibration,
    detect_bins_in_frame,
    make_bin_entry,
    save_bin_coordinates,
    draw_text_jp,
    draw_split_line,
    open_camera,
    parse_where,
    ARM_RIGHT,
    ARM_LEFT,
    ARM_NAMES,
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
ARMS = (ARM_RIGHT, ARM_LEFT)   # 1: 右腕(下) / 2: 左腕(上)

# アームごとの ASプログラム名・ポーズ変数名
#   移動  : POINT {arm}: {pose} = TRANS(...) → EXECUTE {arm}:{goto}
#   ハンド: EXECUTE {arm}:{pump_on} / {pump_off}  (air.py の pump_on / pump_off を使用)
# ※ アーム2で動かない場合は、ここの名前をコントローラー側に合わせて変えるだけでOK
ARM_PROGRAMS = {
    ARM_RIGHT: {"pose": "p1", "goto": "goto1", "pump_on": "pump_on", "pump_off": "pump_off"},
    ARM_LEFT:  {"pose": "p1", "goto": "goto1", "pump_on": "pump_on", "pump_off": "pump_off"},
}

# カメラ番号 (環境に合わせて変更)
CAMERA_INDEX = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 高さ設定 (単位: mm / アームごと)
# ・平面的(X, Y)移動時: Z = 430.0 mm で完全固定（絶対に机や箱にぶつからない安全上空）
# ・机の表面限界: Z = 294.0 mm（これより下へは絶対に下げない衝突防止ライン）
# ※ 上アームと下アームで取り付け高さが違い、机のZが異なる場合はアームごとに値を変える。
#    camera.py の Mode 1 で表示される「目印に合わせた時の先端Z」を参考にすること。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARM_Z = {
    ARM_RIGHT: {
        "table_limit": 294.0,   # 机の表面限界 (mm) ※これより下(小さい数値)には下げない
        "transit": 430.0,       # ものを持って移動・上空移動する時の固定Z座標 (mm)
        "grasp": 298.0,         # 物体を掴む高さ (mm) ※机面294mmより4mm手前の安全位置
        "bin_drop": 350.0,      # 箱の真上でゴミを離す高さ (mm) ※430mmから垂直に下げて優しく投下
    },
    ARM_LEFT: {
        "table_limit": 294.0,
        "transit": 430.0,
        "grasp": 298.0,
        "bin_drop": 350.0,
    },
}

# 待機・ホーム位置 (安全上空 430mm / アームごと)
# ※ 起動時に WHERE コマンドで実際のアーム初期位置 (X, Y) を自動取得して上書きする。
#    下は万一 WHERE 取得に失敗したときのフォールバック値 (稼働エリア内の左右端)。
HOME_POS = {
    ARM_RIGHT: {"x": 280.0, "y": 320.0, "z": 430.0},
    ARM_LEFT:  {"x": -280.0, "y": 320.0, "z": 430.0},
}

# 校正した範囲の外にある物体をスキップするか (Falseなら警告のみ出して動作自体は実行する)
REACH_CHECK = False

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
    CATEGORY_PAPER:      "青の箱",   # 紙ごみ       -> 青の箱
    CATEGORY_PLASTIC:    "黄の箱",   # プラスチック -> 黄の箱
    CATEGORY_CAN_BOTTLE: "赤の箱",   # 缶ビン       -> 赤の箱
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

# アーム動作中もカメラ映像をリアルタイム更新するためのプレビューコンテキスト
_PREVIEW_CTX = {"cap": None, "window_name": None, "calib": None, "status_text": ""}


def set_preview_context(cap=None, window_name=None, calib=None):
    """アーム動作中にカメラ映像を更新するための情報を登録する"""
    _PREVIEW_CTX["cap"] = cap
    _PREVIEW_CTX["window_name"] = window_name
    _PREVIEW_CTX["calib"] = calib


def pump_camera_preview(status_text=None):
    """
    アーム動作待ちの間にカメラから最新フレームを読み取り、画面を更新する（案A）。
    重いYOLO推論は行わないため動作が重くならず、カメラバッファに古い画像が溜まるのも防ぐ。
    """
    cap = _PREVIEW_CTX.get("cap")
    win = _PREVIEW_CTX.get("window_name")
    calib = _PREVIEW_CTX.get("calib")
    if cap is None or win is None:
        return
    if status_text is not None:
        _PREVIEW_CTX["status_text"] = status_text
    ret, frame = cap.read()
    if not ret:
        return
    if calib is not None:
        draw_split_line(frame, calib)
    msg = _PREVIEW_CTX.get("status_text") or "ARM MOVING..."
    draw_text_jp(frame, f"【動作中】 {msg}", (20, 25), font_size=18, color=(0, 165, 255))
    cv2.imshow(win, frame)
    cv2.waitKey(1)


def reset_error_and_motor_on(sock, arm):
    """異常停止時にエラーリセット(ERESET)とモータ再ONを行って次ターンが詰まるのを防ぐ"""
    print(f"[INFO] エラー復旧: ERESET -> ZPOWER {arm}: ON を実行します")
    send_cmd(sock, "ERESET", wait=0.8)
    send_cmd(sock, f"ZPOWER {arm}: ON", wait=1.0)


def wait_for_completion(sock, arm=None, max_wait=30):
    """
    duAroが完了メッセージを返すまで待つ（待機中もカメラ映像をリアルタイム更新）。
    エラーを検知した場合は自動でエラーリセットを行い False を返す。
    """
    start = time.time()
    full_resp = ""
    sock.settimeout(0.05)
    while time.time() - start < max_wait:
        pump_camera_preview()
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
                if arm is not None:
                    reset_error_and_motor_on(sock, arm)
                return False
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[ERROR] 受信エラー: {e}")
            return False
    print(f"[WARN] {max_wait}秒以内に完了メッセージを検知できませんでした")
    if arm is not None:
        reset_error_and_motor_on(sock, arm)
    return False


def motor_on(sock, arm):
    """モータ電源をONにする"""
    print(f"[INFO] モータ電源ON: ZPOWER {arm}: ON")
    resp = send_cmd(sock, f"ZPOWER {arm}: ON", wait=1.0)
    print(f"[RECV] {resp.strip()}")
    return resp


def record_initial_home_positions(sock):
    """
    起動時の各アームの現在位置 (X, Y) を WHERE コマンドで取得し、
    初期位置 (HOME_POS) として記録する。
    これにより、動作範囲外の仮座標へ移動して異常停止するのを防ぐ。
    """
    print("\n[INFO] 起動時のアーム位置を初期位置(HOME_POS)として取得しています...")
    for arm in ARMS:
        try:
            text = send_cmd(sock, f"WHERE {arm}:", wait=0.5)
            send_cmd(sock, "", wait=0.1)  # 連続表示モード解除用の空Enter
            x, y, z, _o, _a, _t = parse_where(text)
            transit_z = ARM_Z[arm]["transit"]
            HOME_POS[arm] = {"x": round(x, 2), "y": round(y, 2), "z": transit_z}
            print(f"[INFO]   {ARM_NAMES[arm]} 初期位置を登録: "
                  f"X={x:.1f}, Y={y:.1f}, Z={transit_z:.1f} (現在Z={z:.1f})")
        except Exception as e:
            fb = HOME_POS[arm]
            print(f"[WARN]   {ARM_NAMES[arm]} の現在位置取得に失敗 ({e}) -> "
                  f"フォールバック値 (X={fb['x']:.1f}, Y={fb['y']:.1f}, Z={fb['z']:.1f}) を使用します")


def set_point(sock, arm, x, y, z, o=0.0, a=0.0, t=0.0):
    """
    POINT指令 + TRANS関数で座標を設定する。
    「変更？」プロンプトに空Enter で確定。
    """
    pose = ARM_PROGRAMS[arm]["pose"]
    cmd = (
        f"POINT {arm}: {pose} = "
        f"TRANS({x:.2f},{y:.2f},{z:.2f},{o:.2f},{a:.2f},{t:.2f})"
    )
    resp1 = send_cmd(sock, cmd, wait=0.5)

    # 「変更？」が届くまで最大3秒待つ
    deadline = time.time() + 3.0
    while "変更" not in resp1 and time.time() < deadline:
        pump_camera_preview()
        time.sleep(0.1)
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


def move_to_xyz(sock, arm, x, y, z):
    """アーム arm を指定座標 (x, y, z) へ移動して完了を待つ"""
    # 【絶対安全ガード】机面以下への下降を阻止するリミッター (アームごと)
    limit = ARM_Z[arm]["table_limit"]
    if z < limit:
        print(f"[SAFETY GUARD] 指令Z={z:.1f} は机面({limit}mm)より下のため、安全限界 Z={limit:.1f} に自動制限しました！")
        z = limit

    msg = f"{ARM_NAMES[arm]} 移動: X={x:.1f}, Y={y:.1f}, Z={z:.1f}"
    print(f"[INFO] {msg}")
    pump_camera_preview(msg)
    set_point(sock, arm, x, y, z)

    # EXECUTE でプログラム実行
    goto = ARM_PROGRAMS[arm]["goto"]
    try:
        flush_buffer(sock)
        sock.settimeout(2.0)
        sock.sendall(f"EXECUTE {arm}:{goto}\r\n".encode("shift_jis"))
    except Exception as e:
        print(f"[ERROR] EXECUTE 送信エラー: {e}")
        return False

    return wait_for_completion(sock, arm=arm)


def gripper_control(sock, arm, grasp=True):
    """ハンドの開閉（空気ポンプ: pump_on / pump_off）"""
    progs = ARM_PROGRAMS[arm]
    if grasp:
        msg = f"{ARM_NAMES[arm]} ハンド閉じる(掴む): {progs['pump_on']}"
        print(f"[INFO] {msg}")
        pump_camera_preview(msg)
        success, _ = pump_on(sock, arm=arm, program_name=progs["pump_on"])
    else:
        msg = f"{ARM_NAMES[arm]} ハンド開く(離す): {progs['pump_off']}"
        print(f"[INFO] {msg}")
        pump_camera_preview(msg)
        success, _ = pump_off(sock, arm=arm, program_name=progs["pump_off"])
    if not success:
        print("[WARN] ハンド開閉コマンドの完了確認に失敗しました")
        reset_error_and_motor_on(sock, arm)
    return success


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.5. 双腕の待機管理 (動かすアームは常に1本だけ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def go_home(sock, arm, arm_state):
    """アームを初期位置 (安全上空 Z=430) へ移動する"""
    home = HOME_POS[arm]
    print(f">> {ARM_NAMES[arm]} を初期位置へ復帰 (X={home['x']:.1f}, Y={home['y']:.1f}, Z={home['z']:.1f})")
    ok = move_to_xyz(sock, arm, home["x"], home["y"], home["z"])
    arm_state[arm] = ok
    return ok


def return_all_arms_home(sock, arm_state):
    """初期位置に戻っていないアームをすべて初期位置へ戻す"""
    for arm in ARMS:
        if not arm_state.get(arm, False):
            go_home(sock, arm, arm_state)


def ensure_other_arm_home(sock, arm, arm_state):
    """これから arm を動かす前に、もう片方のアームが待機位置にいることを保証する"""
    other = ARM_LEFT if arm == ARM_RIGHT else ARM_RIGHT
    if arm_state.get(other):
        return True
    print(f"[INFO] 衝突防止: 先に {ARM_NAMES[other]} を待機位置へ退避させます")
    if not go_home(sock, other, arm_state):
        print(f"[ERROR] {ARM_NAMES[other]} の退避に失敗したため、{ARM_NAMES[arm]} は動かしません")
        return False
    return True


def choose_arm(calib, px, py):
    """
    物体のピクセル位置から担当アームを決める。
    戻り値: (arm, rx, ry, reachable)
    """
    arm = calib.arm_for_pixel(px)
    rx, ry = calib.pixel_to_robot(px, py, arm)
    reachable = calib.in_range(arm, px, py)
    return arm, rx, ry, reachable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ゴミ箱自動検出（起動時に実行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def auto_detect_bins(cap, calib):
    """
    カメラ映像から赤・黄・青のゴミ箱を色で自動検出し、
    各アームから見たロボット座標 (X, Y) を返す。

    複数フレームの検出結果を平均化してノイズを低減する。
    戻り値: {"赤の箱": {"pixel": [u, v], "arm1": {"x","y","in_range"}, "arm2": {...}}, ...}
    """
    print("\n[INFO] ゴミ箱を自動検出しています...")
    print("[INFO] 赤・黄・青のゴミ箱がカメラに映っていることを確認してください")

    # カメラの露出が安定するまで読み飛ばす
    for _ in range(30):
        cap.read()

    # 複数フレームで検出し、中央値を取る（安定化）
    NUM_SAMPLES = 10
    samples = {name: [] for name in ["赤の箱", "黄の箱", "青の箱"]}

    for i in range(NUM_SAMPLES):
        ret, frame = cap.read()
        if not ret:
            continue
        centers, _ = detect_bins_in_frame(frame)
        for bin_name, center in centers.items():
            if center is not None:
                samples[bin_name].append(center)
        time.sleep(0.05)

    # 各箱の中央値を計算 → 両アームのロボット座標に変換
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

        entry = make_bin_entry((px_median, py_median), calib)
        bin_coords[bin_name] = entry
        a1, a2 = entry["arm1"], entry["arm2"]
        print(f"[INFO] {bin_name} (pixel: {px_median}, {py_median}): "
              f"アーム1=({a1['x']:.1f}, {a1['y']:.1f}) / アーム2=({a2['x']:.1f}, {a2['y']:.1f})")
        for arm in ARMS:
            if not entry[f"arm{arm}"]["in_range"]:
                print(f"[WARN]   {bin_name} は {ARM_NAMES[arm]} の校正範囲外です。"
                      "箱を中央 (中央の目印付近) に寄せることをおすすめします")

    return bin_coords


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 分別ピッキングシーケンス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def execute_sorting(sock, target_info, bin_coords, calib, arm_state):
    """
    検出された物体を、担当アームで掴み、対応する箱へ運んで捨てる。

    sock        : duAroソケット
    target_info : WasteDetector.detect() の戻り値
    bin_coords  : {"赤の箱": {"pixel": [...], "arm1": {"x","y"}, "arm2": {"x","y"}}, ...}
    calib       : camera.Calibration (アームごとのホモグラフィ・左右の境界)
    arm_state   : {1: 待機位置にいるか, 2: 待機位置にいるか}
    """
    category = target_info["category"]
    box_name = CATEGORY_TO_BIN.get(category)

    if not box_name:
        print(f"[WARN] カテゴリ '{category}' に対応する箱が定義されていません")
        return False

    if box_name not in bin_coords:
        print(f"[WARN] {box_name} の座標が取得できていません（未検出）")
        return False

    # 物体のピクセル座標 → 担当アーム・そのアームのロボット座標
    px, py = target_info["pixel"]
    arm, rx, ry, reachable = choose_arm(calib, px, py)
    if not reachable:
        if REACH_CHECK:
            print(f"[WARN] 物体 (pixel: {px}, {py}) は {ARM_NAMES[arm]} の校正範囲外のためスキップします")
            print("       キャリブレーションの目印の内側に物体を置いてください")
            return False
        else:
            print(f"[WARN] 物体 (pixel: {px}, {py}) は {ARM_NAMES[arm]} の校正範囲より少し外側ですが、そのまま実行します")

    bin_pos = bin_coords[box_name][f"arm{arm}"]
    z = ARM_Z[arm]
    transit, grasp_z, drop_z = z["transit"], z["grasp"], z["bin_drop"]

    print("\n" + "=" * 50)
    print(f"【分別開始】 対象: {target_info['class']} ({category})  担当: {ARM_NAMES[arm]}")
    print(f"  ・物体位置   : X={rx:.1f}, Y={ry:.1f}, Z={grasp_z:.1f} mm (机面限界: {z['table_limit']}mm)")
    print(f"  ・平面移動高 : Z={transit:.1f} mm (固定安全上空)")
    print(f"  ・搬送先     : {box_name} (X={bin_pos['x']:.1f}, Y={bin_pos['y']:.1f})")
    print("=" * 50)

    # --- 0. もう片方のアームを待機位置へ (衝突防止) ---
    if not ensure_other_arm_home(sock, arm, arm_state):
        return False
    arm_state[arm] = False   # これから動かすので「待機位置にいない」扱い

    # --- 1. ピック動作（掴む） ---
    # ① まずハンドを開く
    gripper_control(sock, arm, grasp=False)

    # ② 平面移動: 安全上空 Z=430mm で物体の真上へ移動
    print(f">> [1/7] 物体上空へ平面移動 (安全高さ Z={transit:.1f})")
    if not move_to_xyz(sock, arm, rx, ry, transit):
        return False

    # ③ 真下に下降して掴む高さへ
    print(f">> [2/7] 垂直下降 (掴み位置 Z={grasp_z:.1f})")
    if not move_to_xyz(sock, arm, rx, ry, grasp_z):
        return False

    # ④ チャックを閉じて掴む
    print(">> [3/7] 掴む")
    gripper_control(sock, arm, grasp=True)

    # ⑤ 安全上空 Z=430mm へ真上に持ち上げる
    print(f">> [4/7] 真上に持ち上げ (安全高さ Z={transit:.1f})")
    if not move_to_xyz(sock, arm, rx, ry, transit):
        return False

    # --- 2. プレイス動作（箱へ捨てる） ---
    # ⑥ 平面移動: 安全上空 Z=430mm で箱の上空へ移動
    print(f">> [5/7] {box_name}の上空へ平面移動 (安全高さ Z={transit:.1f})")
    if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], transit):
        return False

    # ⑦ 箱の真上で投下高さへ垂直下降 (drop_z が transit と異なる場合のみ)
    if abs(drop_z - transit) > 0.5:
        print(f">> [6/7] 投下高さへ垂直下降 (Z={drop_z:.1f})")
        if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], drop_z):
            go_home(sock, arm, arm_state)
            return False

    # ⑧ ハンドを開いてゴミを落とす
    print(f">> [7/7] 投下（{box_name}に捨てる）")
    gripper_control(sock, arm, grasp=False)

    # ⑨ 安全上空 Z=430mm へ垂直上昇 (drop_z から持ち上げてから横移動する)
    if abs(drop_z - transit) > 0.5:
        print(f">> 退避上昇 (安全高さ Z={transit:.1f})")
        if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], transit):
            go_home(sock, arm, arm_state)
            return False

    # ⑩ 初期位置へ平面移動 (Z=430)
    print(f">> 初期位置へ復帰 (安全高さ Z={transit:.1f})")
    if not go_home(sock, arm, arm_state):
        print("[WARN] 初期位置への復帰に失敗しました")
        return False

    print("=== 分別完了 ===\n")
    return True


def draw_target_guide(display_frame, target, calib, y_text):
    """検出対象・担当アーム・ロボット座標を画面に表示する"""
    cat = target["category"]
    box_name = CATEGORY_TO_BIN.get(cat, "未設定")
    px, py = target["pixel"]
    arm, rx, ry, reachable = choose_arm(calib, px, py)
    reach = "" if reachable else " 範囲外!"
    color = (0, 255, 0) if reachable else (0, 0, 255)
    guide = (
        f"TARGET: {cat} -> [{box_name}] {ARM_NAMES[arm]} "
        f"(conf:{target['conf']:.2f}, X={rx:.0f}, Y={ry:.0f}){reach}"
    )
    draw_text_jp(display_frame, guide, (20, y_text), font_size=18, color=color)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.5. 全自動分別ループ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def auto_sorting_loop(sock, cap, detector, bin_coords, calib, arm_state, window_name):
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
        draw_split_line(display_frame, calib)

        # 画面にステータス表示
        status = f"AUTO MODE | Sorted: {sorted_count} | Scanning: {no_detect_count}/{AUTO_NO_DETECT_LIMIT}"
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if target:
            draw_target_guide(display_frame, target, calib, 65)
        else:
            cv2.putText(display_frame, "Scanning for waste...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        cv2.imshow(window_name, display_frame)

        # 'q' キーで緊急停止 (ウィンドウ・ターミナル両方対応)
        key = get_pressed_key(wait_ms=1)
        if key == ord("q"):
            print("\n[INFO] 'q' キーを検知: 全自動モードを緊急停止しました")
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
        success = execute_sorting(sock, target, bin_coords, calib, arm_state)

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

    # 全完了時に初期位置へ戻っていないアームがあれば初期位置へ戻す
    return_all_arms_home(sock, arm_state)

    print("\n" + "=" * 50)
    print(f"【全自動分別完了】 合計 {sorted_count} 個のゴミを分別しました（初期位置復帰済み）")
    print("=" * 50 + "\n")

    return sorted_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. メインループ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # --- A. キャリブレーション読み込み ---
    try:
        calib = load_calibration()
        left_arm = calib.arm_for_pixel(calib.split_u - 1)
        right_arm = calib.arm_for_pixel(calib.split_u + 1)
        print(f"[INFO] キャリブレーション読み込み完了 "
              f"(左右の境界 split_u={calib.split_u:.0f}px: "
              f"画面左->{ARM_NAMES[left_arm]} / 画面右->{ARM_NAMES[right_arm]})")
        for arm in ARMS:
            mz = calib.measured_z.get(f"arm{arm}")
            if mz is not None:
                print(f"[INFO]   {ARM_NAMES[arm]}: 校正時の先端Z={mz:.1f} mm / "
                      f"掴む高さ={ARM_Z[arm]['grasp']:.1f} / 机面限界={ARM_Z[arm]['table_limit']:.1f}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e)
        print("[INFO] 先に camera.py の Mode 1 (双腕6点キャリブレーション) を実行してください")
        raise SystemExit(1)

    # --- B. カメラ起動 (WindowsではDSHOWを優先してMSMFエラー -1072873822 を防止) ---
    try:
        cap = open_camera(CAMERA_INDEX, warmup_frames=0)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    # --- C. ゴミ箱を自動検出 ---
    bin_coords = auto_detect_bins(cap, calib)

    if not bin_coords:
        print("[ERROR] ゴミ箱が1つも検出できませんでした")
        print("[INFO] カメラに赤・黄・青の箱が映るように配置してから再実行してください")
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

    # --- E. モータON (両アーム) & 起動時の初期位置(HOME_POS)を自動取得 ---
    for arm in ARMS:
        motor_on(sock, arm)
    record_initial_home_positions(sock)

    # 各アームが待機位置にいるか (起動時は初期位置にいるので True)
    arm_state = {arm: True for arm in ARMS}

    # --- F. YOLO検出器の初期化 ---
    detector = WasteDetector()

    # --- G. メインループ ---
    window_name = "duAro Waste Sorting [s: Sort, a: Auto, q: Quit]"
    cv2.namedWindow(window_name)
    set_preview_context(cap=cap, window_name=window_name, calib=calib)

    print("\n" + "=" * 50)
    print("【duAro ゴミ自動分別システム (双腕)】")
    print("=" * 50)
    print(f"  検出済みゴミ箱: {', '.join(bin_coords.keys())}")
    print(f"  担当: 画面左側 -> {ARM_NAMES[left_arm]} / 画面右側 -> {ARM_NAMES[right_arm]}")
    print()
    print("  [操作方法]")
    print("  ・カメラの視野にゴミを置く")
    print("  ・'s' キー → 1個だけ分別（手動モード）")
    print("  ・'a' キー → 全自動分別（ゴミがなくなるまで繰り返す）")
    print("  ・'q' キー → 終了")
    print("=" * 50 + "\n")

    # duAro側の前提プログラム確認メッセージ
    print(f"[INFO] duAro側に以下のASプログラムが登録されている必要があります (両アームで実行):")
    for arm in ARMS:
        p = ARM_PROGRAMS[arm]
        print(f"       {ARM_NAMES[arm]}: EXECUTE {arm}:{p['goto']} (JMOVE {p['pose']}) / "
              f"{p['pump_on']} (OPENI) / {p['pump_off']} (CLOSEI)")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] カメラからフレームを取得できませんでした")
                break

            # YOLO で物体検出・カテゴリ判定
            target, display_frame = detector.detect(frame)
            draw_split_line(display_frame, calib)

            # 画面上に案内を表示 (担当アーム・ロボット座標つき)
            if target:
                draw_target_guide(display_frame, target, calib, 20)
                cv2.putText(display_frame, "'s': Sort 1 / 'a': Auto-sort all", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No target detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            cv2.imshow(window_name, display_frame)
            key = get_pressed_key(wait_ms=20)

            if key == ord("q"):
                print("[INFO] 'q' キーを検知: プログラムを終了します")
                break

            elif key == ord("s"):
                print("[INFO] 's' キーを検知しました (手動1個分別)")
                if target is None:
                    print("[WARN] カメラの視野内にゴミが検出されていません！カメラの前にゴミを置いてください")
                    continue

                # 分別先の箱が検出済みか確認
                cat = target["category"]
                needed_bin = CATEGORY_TO_BIN.get(cat)
                if needed_bin and needed_bin not in bin_coords:
                    print(f"[WARN] {needed_bin} が検出されていないため分別できません")
                    continue

                execute_sorting(sock, target, bin_coords, calib, arm_state)

            elif key == ord("a"):
                print("[INFO] 'a' キーを検知しました (全自動分別開始)")
                auto_sorting_loop(sock, cap, detector, bin_coords, calib, arm_state, window_name)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C で中断されました")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sock:
            disconnect(sock)
        print("[INFO] ロボットから切断しました")
