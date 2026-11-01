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
            elif char in ("b", "ｂ"):
                return ord("b")
            elif char in ("1", "１"):
                return ord("1")
            elif char in ("2", "２"):
                return ord("2")
            elif char in ("3", "３"):
                return ord("3")
            elif char in ("q", "ｑ", "\x1b"):
                return ord("q")
        except Exception:
            pass

    # 2. OpenCV ウィンドウ側でのキー判定
    if key in (ord("s"), ord("S")):
        return ord("s")
    elif key in (ord("a"), ord("A")):
        return ord("a")
    elif key in (ord("b"), ord("B")):
        return ord("b")
    elif key in (ord("1"), ord("2"), ord("3")):
        return key
    elif key in (ord("q"), ord("Q"), 27):
        return ord("q")

    return key

# --- 自作モジュール ---
from camera import (
    load_calibration,
    load_bin_coordinates,
    make_bin_entry,
    save_bin_coordinates,
    draw_text_jp,
    draw_split_line,
    open_camera,
    parse_where,
    BIN_COLOR_RANGES,
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
# ・平面的(X, Y)移動時: Z = 500.0 mm で完全固定（絶対に机や箱にぶつからない安全上空）
# ・上空の上限: Z = 530.0 mm（これより上へは上げない上限ライン）
# ・机の表面限界: Z = 260.0 mm（これより下へは絶対に下げない衝突防止ライン）
# ※ 上アームと下アームで取り付け高さが違い、机のZが異なる場合はアームごとに値を変える。
#    camera.py の Mode 1 で表示される「目印に合わせた時の先端Z」を参考にすること。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARM_Z = {
    ARM_RIGHT: {
        "table_limit": 260.0,   # 机の表面限界 (mm) ※これより下(小さい数値)には下げない
        "upper_limit": 530.0,   # 上空の限界 (mm) ※これより上(大きい数値)には上げない
        "transit": 500.0,       # ものを持って移動・上空移動する時の固定Z座標 (mm)
        "grasp": 298.0,         # 物体を掴む高さ (mm) ※デフォルト(紙コップ用)
        "bin_drop": 350.0,      # 箱の真上でゴミを離す高さ (mm)
    },
    ARM_LEFT: {
        "table_limit": 260.0,
        "upper_limit": 530.0,
        "transit": 500.0,
        "grasp": 298.0,
        "bin_drop": 350.0,
    },
}

# 待機・ホーム位置 (安全上空 500mm / アームごと)
# ※ 起動時に WHERE コマンドで実際のアーム初期位置 (X, Y) を自動取得して上書きする。
#    下は万一 WHERE 取得に失敗したときのフォールバック値 (稼働エリア内の左右端)。
HOME_POS = {
    ARM_RIGHT: {"x": 280.0, "y": 320.0, "z": 500.0},
    ARM_LEFT:  {"x": -280.0, "y": 320.0, "z": 500.0},
}

# 校正した範囲の外にある物体をスキップするか (Falseなら警告のみ出して動作自体は実行する)
REACH_CHECK = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.5. 速度設定 (単位: % / モニタ速度 1〜100%)
# ・通常速度 (SPEED_NORMAL) を基準に、空荷の移動などを約1.5倍 (SPEED_FAST) に高速化する。
# ※ duAroのモニタ速度の上限は 100% です（例: 通常 65% → 高速 100% で約1.5倍）。
#    もし現在もっとゆっくり動かしている場合（例: 40%）は、通常=40, 運搬=50, 高速=60 などに調整してください。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPEED_NORMAL = 30    # 慎重に動くとき (下降して掴む・持ち上げる・箱へ降りるとき) [%]
SPEED_CARRY  = 30     # ゴミを持って箱の上空へ水平移動するとき [%]
SPEED_FAST   = 30    # 空荷で移動するとき (物体上空へ行くとき・捨てた後に初期位置へ帰るとき: 約1.5倍) [%]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全自動モードの設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTO_NO_DETECT_LIMIT = 5    # 連続未検出回数がこの値に達したら「全完了」と判断
AUTO_STABILIZE_WAIT = 1.0   # 分別後、次の検出前の待機時間（秒）
AUTO_MAX_FAILURES = 3       # 連続失敗がこの値に達したら自動停止


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 分別マッピング & カテゴリ別の掴む高さ (単位: mm)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY_TO_BIN = {
    CATEGORY_PAPER:      "青の箱",   # 紙ごみ       -> 青の箱
    CATEGORY_PLASTIC:    "黄の箱",   # プラスチック -> 黄の箱
    CATEGORY_CAN_BOTTLE: "赤の箱",   # 缶ビン       -> 赤の箱
}

# カテゴリごとの掴む高さ Z (mm)。未指定のカテゴリは ARM_Z[arm]["grasp"] (298.0mm) を使用
CATEGORY_GRASP_Z = {
    CATEGORY_PAPER:      298.0,      # 紙コップなど (紙ごみ) -> 298mmで掴む
    CATEGORY_PLASTIC:    264.0,      # プラスチック -> 264mmで掴む
    CATEGORY_CAN_BOTTLE: 390.0,      # ペットボトルなど (缶ビン) -> 390mmで掴む
}

# カテゴリごとの箱でおろす高さ (投下高さ) Z (mm)。
# 紙コップ(350mm)に対し、ペットボトル(缶ビン)は100mm上の 450.0mm でおろす
CATEGORY_DROP_Z = {
    CATEGORY_PAPER:      350.0,      # 紙コップなど (紙ごみ)
    CATEGORY_PLASTIC:    350.0,      # プラスチック
    CATEGORY_CAN_BOTTLE: 450.0,      # ペットボトルなど (缶ビン) -> 紙コップより100mm上でおろす
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
_CURRENT_SPEED = {"value": None}
# 各アームの現在座標 (X, Y, Z) を保持し、X軸・Y軸・Z軸を1軸ずつ順番に動かすために使用
_ARM_POS = {}


def set_robot_speed(sock, speed_pct, arm=None):
    """
    duAroのモニタ速度 (1〜100%) を変更する。
    同じ速度が連続する場合は通信を省略して無駄な待ち時間をなくす。
    """
    speed_pct = max(1, min(100, int(round(speed_pct))))
    if _CURRENT_SPEED["value"] == speed_pct:
        return
    # アーム指定の SPEED と全体 SPEED の両対応
    if arm is not None:
        resp = send_cmd(sock, f"SPEED {arm}: {speed_pct}", wait=0.15)
        if _ERROR_PATTERN.search(resp):
            send_cmd(sock, f"SPEED {speed_pct}", wait=0.15)
    else:
        send_cmd(sock, f"SPEED {speed_pct}", wait=0.15)
    _CURRENT_SPEED["value"] = speed_pct


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


def query_arm_xyz(sock, arm, retries=3):
    """WHERE コマンドでアームの実際の現在位置 (x, y, z) を取得して _ARM_POS を更新する"""
    for attempt in range(retries):
        try:
            text = send_cmd(sock, f"WHERE {arm}:", wait=0.5)
            send_cmd(sock, "", wait=0.15)  # 連続表示モード解除用の空Enter
            x, y, z, _o, _a, _t = parse_where(text)
            _ARM_POS[arm] = {"x": float(x), "y": float(y), "z": float(z)}
            return float(x), float(y), float(z)
        except Exception:
            time.sleep(0.3)
    return None


def record_initial_home_positions(sock):
    """
    起動時の各アームの現在位置 (X, Y, Z すべて) を WHERE コマンドで取得し、
    初期位置 (HOME_POS) および現在座標 (_ARM_POS) として記録する。
    """
    print("\n[INFO] 起動時のアーム位置(X, Y, Z)を初期位置(HOME_POS)として取得しています...")
    for arm in ARMS:
        pos = query_arm_xyz(sock, arm, retries=3)
        if pos is not None:
            x, y, z = pos
            # 初期位置の X, Y, Z すべてをそのまま記憶する (最低安全限界以上のZ)
            safe_home_z = max(round(z, 2), ARM_Z[arm]["table_limit"])
            HOME_POS[arm] = {"x": round(x, 2), "y": round(y, 2), "z": safe_home_z}
            print(f"[INFO]   {ARM_NAMES[arm]} 初期位置(X,Y,Z)を登録: "
                  f"X={x:.1f}, Y={y:.1f}, Z={safe_home_z:.1f}")
        else:
            fb = HOME_POS[arm]
            _ARM_POS[arm] = {"x": float(fb["x"]), "y": float(fb["y"]), "z": float(fb["z"])}
            print(f"[WARN]   {ARM_NAMES[arm]} の現在位置取得に失敗 -> "
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


def _move_single_step(sock, arm, x, y, z, axis_label=""):
    """1ステップ分の座標設定とEXECUTE実行を行い、完了したら _ARM_POS を更新する"""
    limit = ARM_Z[arm]["table_limit"]
    upper = ARM_Z[arm]["upper_limit"]
    if z < limit:
        print(f"[SAFETY GUARD] 指令Z={z:.1f} は机面({limit}mm)より下のため、安全限界 Z={limit:.1f} に自動制限しました！")
        z = limit
    elif z > upper:
        print(f"[SAFETY GUARD] 指令Z={z:.1f} は上空限界({upper}mm)より上のため、上限 Z={upper:.1f} に自動制限しました！")
        z = upper

    spd_str = f" (速度:{_CURRENT_SPEED['value']}%)" if _CURRENT_SPEED["value"] else ""
    axis_str = f"[{axis_label}] " if axis_label else ""
    msg = f"{ARM_NAMES[arm]} {axis_str}移動: X={x:.1f}, Y={y:.1f}, Z={z:.1f}{spd_str}"
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

    ok = wait_for_completion(sock, arm=arm)
    if ok:
        _ARM_POS[arm] = {"x": float(x), "y": float(y), "z": float(z)}
    return ok


def move_to_xyz(sock, arm, x, y, z, speed=None):
    """
    Z軸と水平(XY)の同時移動を完全に禁止し、水平(XY)移動とZ軸(上下)移動を必ず分けて (x, y, z) へ移動する。
      1) 横移動(XまたはY)がある場合で現在Zが目標Zより低いときは、
         まずその場で【Z軸だけ】目標Zへ垂直に上げる。
      2) 水平移動は高さ(Z)を固定したまま、【X, Y 同時（斜め移動可）】で (x, y) へ移動する。
      3) 最後に Z軸の移動が必要なら、到着した (x, y) の真上で【Z軸だけ】垂直に動かす。
    """
    limit = ARM_Z[arm]["table_limit"]
    upper = ARM_Z[arm]["upper_limit"]
    if z < limit:
        print(f"[SAFETY GUARD] 指令Z={z:.1f} は机面({limit}mm)より下のため、安全限界 Z={limit:.1f} に自動制限しました！")
        z = limit
    elif z > upper:
        print(f"[SAFETY GUARD] 指令Z={z:.1f} は上空限界({upper}mm)より上のため、上限 Z={upper:.1f} に自動制限しました！")
        z = upper

    if speed is not None:
        set_robot_speed(sock, speed, arm=arm)

    cur = _ARM_POS.get(arm)
    if cur is None:
        query_arm_xyz(sock, arm, retries=2)
        cur = _ARM_POS.get(arm)
    if cur is None:
        # 万一現在位置が取得できなかった場合も、安全上空(transit)でXYへ移動してからZを合わせる
        safe_z = max(ARM_Z[arm]["transit"], z)
        if not _move_single_step(sock, arm, x, y, safe_z, axis_label="XY水平"):
            return False
        if abs(z - safe_z) >= 1.0:
            return _move_single_step(sock, arm, x, y, z, axis_label="Z軸")
        return True

    cx, cy, cz = cur["x"], cur["y"], cur["z"]
    dx = abs(x - cx)
    dy = abs(y - cy)
    MOVE_TOL = 1.0  # 1mm未満の差は同一位置とみなす

    # 1. 横移動(XまたはY)がある場合で、現在Zが目標Zより低いときは【必ず最初にZ軸だけ】上げる
    if (dx >= MOVE_TOL or dy >= MOVE_TOL) and (z > cz + MOVE_TOL):
        if not _move_single_step(sock, arm, cx, cy, z, axis_label="Z軸"):
            return False
        cz = z

    # 2. 水平移動: 高さ(cz)を固定したまま、X軸・Y軸を同時に動かして (x, y) へ移動する
    if dx >= MOVE_TOL or dy >= MOVE_TOL:
        if not _move_single_step(sock, arm, x, y, cz, axis_label="XY水平"):
            return False
        cx, cy = x, y

    # 3. Z軸の移動が残っている場合 (垂直下降・垂直上昇)、最後にその場で【Z軸だけ】動かす
    if abs(z - cz) >= MOVE_TOL:
        if not _move_single_step(sock, arm, cx, cy, z, axis_label="Z軸"):
            return False

    return True


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
    """
    アームをはじめの初期位置 (home_x, home_y, home_z すべて) へ1軸ずつ順番に戻す。
      1) 現在Zが安全上空より低ければ、まずその場で Z軸だけ安全上空へ上げる
      2) 動作範囲内の安全な順番 (Y軸 → X軸) で初期位置の (home_x, home_y) へ移動する
      3) 最後にその場で Z軸だけ初期位置の home_z へ戻す
      ※ 万が一、起動時に手で押し込んだ初期位置自体がソフト上の「動作範囲外」と判定された場合は、
         確実に届く少し内側の安全待機座標へ自動補正して復帰する。
    """
    home = HOME_POS[arm]
    safe_z = max(ARM_Z[arm]["transit"], home["z"])
    print(f">> {ARM_NAMES[arm]} を初期位置(X={home['x']:.1f}, Y={home['y']:.1f}, Z={home['z']:.1f})へ復帰")
    query_arm_xyz(sock, arm, retries=2)
    cur = _ARM_POS.get(arm)
    if cur is not None and cur["z"] < safe_z - 1.0:
        if not move_to_xyz(sock, arm, cur["x"], cur["y"], safe_z, speed=SPEED_FAST):
            arm_state[arm] = False
            return False
    # 安全高さ safe_z で水平移動してから、最後に Z軸を home["z"] に合わせる
    ok = move_to_xyz(sock, arm, home["x"], home["y"], home["z"], speed=SPEED_FAST)
    if not ok:
        # 起動時に手で端まで押し込んだ位置が TRANS(o=0) で動作範囲外になる場合の自動補正
        sign = 1.0 if home["x"] >= 0 else -1.0
        fallback_x = sign * min(abs(home["x"]), 260.0)
        fallback_y = max(min(home["y"], 380.0), 320.0)
        print(f"[INFO] 初期位置が動作範囲ギリギリのため、少し内側の安全待機位置 "
              f"(X={fallback_x:.1f}, Y={fallback_y:.1f}, Z={home['z']:.1f}) へ自動補正して戻ります")
        query_arm_xyz(sock, arm, retries=2)
        ok = move_to_xyz(sock, arm, fallback_x, fallback_y, home["z"], speed=SPEED_FAST)
        if ok:
            HOME_POS[arm] = {"x": fallback_x, "y": fallback_y, "z": home["z"]}
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
    grasp_z = CATEGORY_GRASP_Z.get(category, z["grasp"])
    drop_z = CATEGORY_DROP_Z.get(category, z["bin_drop"])
    # ペットボトル等で投下高さ(450mm)が通常の移動高さ(430mm)より高い場合は、運搬時の上空高さもそれに合わせる
    transit = max(z["transit"], drop_z)

    print("\n" + "=" * 50)
    print(f"【分別開始】 対象: {target_info['class']} ({category})  担当: {ARM_NAMES[arm]}")
    print(f"  ・物体ピクセル: u={px}, v={py}  (境界 split_u={calib.split_u:.0f})")
    print(f"  ・箱ピクセル  : {bin_coords[box_name].get('pixel')}")
    print(f"  ・物体位置   : X={rx:.1f}, Y={ry:.1f}, Z={grasp_z:.1f} mm (机面限界: {z['table_limit']}mm)")
    print(f"  ・平面移動高 : Z={transit:.1f} mm (Z軸は単独垂直移動 / XYは同時水平移動)")
    print(f"  ・搬送先     : {box_name} (X={bin_pos['x']:.1f}, Y={bin_pos['y']:.1f}, 投下Z={drop_z:.1f} mm)")
    print("=" * 50)

    # --- 0. もう片方のアームを待機位置へ (衝突防止) & 担当アームの現在座標を同期 ---
    if not ensure_other_arm_home(sock, arm, arm_state):
        return False
    query_arm_xyz(sock, arm, retries=2)
    arm_state[arm] = False   # これから動かすので「待機位置にいない」扱い

    # --- 1. ピック動作（掴む） ---
    # ① まずハンドを開く
    gripper_control(sock, arm, grasp=False)

    # ② 平面移動 (高速 1.5倍): 空荷で安全上空の物体の真上へ移動
    print(f">> [1/7] 物体上空へ高速移動 (安全高さ Z={transit:.1f}, 速度={SPEED_FAST}%)")
    if not move_to_xyz(sock, arm, rx, ry, transit, speed=SPEED_FAST):
        go_home(sock, arm, arm_state)
        return False

    # ③ 真下に下降して掴む高さへ (通常速度: 物体に優しくアプローチ)
    print(f">> [2/7] 垂直下降 (掴み位置 Z={grasp_z:.1f}, 速度={SPEED_NORMAL}%)")
    if not move_to_xyz(sock, arm, rx, ry, grasp_z, speed=SPEED_NORMAL):
        go_home(sock, arm, arm_state)
        return False

    # ④ チャックを閉じて掴む
    print(">> [3/7] 掴む")
    gripper_control(sock, arm, grasp=True)

    # ⑤ 安全上空へ真上に持ち上げる (通常速度: 掴んだ直後の落下防止)
    print(f">> [4/7] 真上に持ち上げ (安全高さ Z={transit:.1f}, 速度={SPEED_NORMAL}%)")
    if not move_to_xyz(sock, arm, rx, ry, transit, speed=SPEED_NORMAL):
        go_home(sock, arm, arm_state)
        return False

    # --- 2. プレイス動作（箱へ捨てる） ---
    # ⑥ 平面移動 (運搬速度): 安全上空で箱の上空へ移動
    print(f">> [5/7] {box_name}の上空へ平面移動 (安全高さ Z={transit:.1f}, 速度={SPEED_CARRY}%)")
    if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], transit, speed=SPEED_CARRY):
        go_home(sock, arm, arm_state)
        return False

    # ⑦ 箱の真上で投下高さへ垂直下降 (通常速度: 箱の中へ丁寧に降ろす)
    if abs(drop_z - transit) > 0.5:
        print(f">> [6/7] 投下高さへ垂直下降 (Z={drop_z:.1f}, 速度={SPEED_NORMAL}%)")
        if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], drop_z, speed=SPEED_NORMAL):
            go_home(sock, arm, arm_state)
            return False

    # ⑧ ハンドを開いてゴミを落とす
    print(f">> [7/7] 投下（{box_name}に捨てる）")
    gripper_control(sock, arm, grasp=False)

    # ⑨ 安全上空 Z=430mm へ垂直上昇 (高速 1.5倍: 空荷になったので素早く退避)
    if abs(drop_z - transit) > 0.5:
        print(f">> 退避上昇 (安全高さ Z={transit:.1f}, 速度={SPEED_FAST}%)")
        if not move_to_xyz(sock, arm, bin_pos["x"], bin_pos["y"], transit, speed=SPEED_FAST):
            go_home(sock, arm, arm_state)
            return False

    # ⑩ 初期位置へ平面移動 (高速 1.5倍: Z=430)
    print(f">> 初期位置へ高速復帰 (安全高さ Z={transit:.1f}, 速度={SPEED_FAST}%)")
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
    grasp_z = CATEGORY_GRASP_Z.get(cat, ARM_Z[arm]["grasp"])
    reach = "" if reachable else " 範囲外!"
    color = (0, 255, 0) if reachable else (0, 0, 255)
    guide = (
        f"TARGET: {cat} -> [{box_name}] {ARM_NAMES[arm]} "
        f"(conf:{target['conf']:.2f}, X={rx:.0f}, Y={ry:.0f}, Z={grasp_z:.0f}){reach}"
    )
    draw_text_jp(display_frame, guide, (20, y_text), font_size=18, color=color)


def draw_bins_overlay(display_frame, bin_coords, selected_bin_name=None):
    """現在登録されている各ゴミ箱の位置を画面上に丸印とラベルで描画する"""
    bin_order = [("1", "赤の箱"), ("2", "黄の箱"), ("3", "青の箱")]
    for num, bname in bin_order:
        color = BIN_COLOR_RANGES.get(bname, {}).get("display_bgr", (255, 255, 255))
        entry = bin_coords.get(bname)
        if entry and "pixel" in entry:
            px, py = int(entry["pixel"][0]), int(entry["pixel"][1])
            cv2.circle(display_frame, (px, py), 8, color, -1)
            cv2.circle(display_frame, (px, py), 11, (255, 255, 255), 2)
            draw_text_jp(display_frame, f"[{num}]{bname}", (px + 12, max(py - 10, 10)),
                         font_size=15, color=color)
    if selected_bin_name:
        draw_text_jp(
            display_frame,
            f"箱クリック調整: [{selected_bin_name}] (1:赤 2:黄 3:青 + 画面クリックで箱位置を直接指定)",
            (20, display_frame.shape[0] - 52),
            font_size=14,
            color=(255, 255, 0),
        )
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

        # YOLO で物体検出 (ゴミ箱の中にある物体は除外)
        target, display_frame = detector.detect(frame, bin_coords=bin_coords)
        draw_split_line(display_frame, calib)
        draw_bins_overlay(display_frame, bin_coords)

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

    # --- C. ゴミ箱座標の読み込み (camera.py Mode 2 で保存した bin_coordinates.json を使用) ---
    bin_coords = {}
    try:
        saved_bins = load_bin_coordinates()
        for bname, binfo in saved_bins.items():
            if "pixel" in binfo:
                bin_coords[bname] = make_bin_entry(binfo["pixel"], calib)
        if bin_coords:
            print(f"[INFO] 保存済みのゴミ箱座標を読み込みました: {', '.join(bin_coords.keys())}")
    except Exception as e:
        print(f"[WARN] {e}")

    missing_bins = [name for name in CATEGORY_TO_BIN.values() if name not in bin_coords]
    if missing_bins:
        print(f"[WARN] 以下の箱がまだ未登録です: {', '.join(missing_bins)}")
        print("[INFO] camera.py の Mode 2 を実行するか、起動後の画面で【1:赤 / 2:黄 / 3:青】+クリックで登録してください")

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
    window_name = "duAro Waste Sorting [s: Sort, a: Auto, 1/2/3+click: Set Bin, q: Quit]"
    cv2.namedWindow(window_name)
    set_preview_context(cap=cap, window_name=window_name, calib=calib)

    selected_bin_for_click = {"name": "赤の箱"}
    bin_list = ["赤の箱", "黄の箱", "青の箱"]

    def _on_sorting_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bname = selected_bin_for_click["name"]
            entry = make_bin_entry((int(x), int(y)), calib)
            bin_coords[bname] = entry
            save_bin_coordinates(bin_coords)
            a1, a2 = entry["arm1"], entry["arm2"]
            print(f"[INFO] クリックで {bname} の位置を更新しました: pixel=({x}, {y}) -> "
                  f"アーム1=({a1['x']:.1f}, {a1['y']:.1f}) / アーム2=({a2['x']:.1f}, {a2['y']:.1f})")
            idx = (bin_list.index(bname) + 1) % len(bin_list)
            selected_bin_for_click["name"] = bin_list[idx]

    cv2.setMouseCallback(window_name, _on_sorting_mouse)

    print("\n" + "=" * 50)
    print("【duAro ゴミ自動分別システム (双腕)】")
    print("=" * 50)
    print(f"  検出済みゴミ箱: {', '.join(bin_coords.keys()) if bin_coords else 'なし(1/2/3キー+画面クリックで設定可)'}")
    print(f"  担当: 画面左側 -> {ARM_NAMES[left_arm]} / 画面右側 -> {ARM_NAMES[right_arm]}")
    print()
    print("  [操作方法]")
    print("  ・カメラの視野にゴミを置く")
    print("  ・'s' キー → 1個だけ分別（手動モード）")
    print("  ・'a' キー → 全自動分別（ゴミがなくなるまで繰り返す）")
    print("  ・'1'/'2'/'3' キー + 画面上の箱をクリック → 赤/黄/青の箱の位置を手動で直接指定")
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

            # YOLO で物体検出・カテゴリ判定 (ゴミ箱の中にある物体は除外)
            target, display_frame = detector.detect(frame, bin_coords=bin_coords)
            draw_split_line(display_frame, calib)
            draw_bins_overlay(display_frame, bin_coords, selected_bin_for_click["name"])

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

            elif key == ord("1"):
                selected_bin_for_click["name"] = "赤の箱"
                print("[INFO] クリック対象を [赤の箱] に設定しました。カメラ画面上の赤い箱をクリックしてください")

            elif key == ord("2"):
                selected_bin_for_click["name"] = "黄の箱"
                print("[INFO] クリック対象を [黄の箱] に設定しました。カメラ画面上の黄色い箱をクリックしてください")

            elif key == ord("3"):
                selected_bin_for_click["name"] = "青の箱"
                print("[INFO] クリック対象を [青の箱] に設定しました。カメラ画面上の青い箱をクリックしてください")

            elif key == ord("s"):
                print("[INFO] 's' キーを検知しました (手動1個分別)")
                if target is None:
                    print("[WARN] カメラの視野内にゴミが検出されていません！カメラの前にゴミを置いてください")
                    continue

                # 分別先の箱が検出済みか確認
                cat = target["category"]
                needed_bin = CATEGORY_TO_BIN.get(cat)
                if needed_bin and needed_bin not in bin_coords:
                    print(f"[WARN] {needed_bin} が未登録です！キーボードの 1(赤)/2(黄)/3(青) を押して画面上の箱をクリックしてください")
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
