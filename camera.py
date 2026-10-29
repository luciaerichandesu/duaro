"""
camera.py — duAro 双腕キャリブレーション & ゴミ箱検出モジュール

【アームの割り当て】(カメラ画面で見て)
    画面の右側 -> アーム1 (下に付いている右腕)
    画面の左側 -> アーム2 (上に付いている左腕)
    左右の境界は、キャリブレーションで使う「中央の目印2点」のピクセルX座標の平均。

Mode 1: 双腕 6点キャリブレーション (座標の手打ちなし)
    机の上の目印を 6 か所 (右2・中央2・左2) 使い、
      ・右2点  : アーム1(右腕) のみ
      ・中央2点: アーム1 と アーム2 の両方 (同じ目印に両アームを合わせる)
      ・左2点  : アーム2(左腕) のみ
    の合計 8 回、duAro に Telnet で `WHERE n:` を送って X, Y, Z を自動取得し、
    アーム1用 H1 / アーム2用 H2 (ピクセル -> ロボットXY) を求めて
    camera_calibration.json に保存する。
    ※ アーム2はアーム1と座標系が異なる (鏡像) ので、アームごとに別の行列が必要。
    ※ WHERE で取れた Z (目印に合わせた時の先端高さ) もアームごとに記録する (確認用)。

Mode 2: ゴミ箱の色検出 (赤・黄・青)
    箱のピクセル位置と、各アームから見たロボット座標を bin_coordinates.json に保存する。

【キー入力について】
    OpenCV ウィンドウ / ターミナル のどちらで押しても反応する。
    IME が全角のままでも (ｓ ｒ ｑ など) 反応する。

必要ライブラリ:
    pip install opencv-python numpy pillow
"""

import json
import os
import re
import socket
import time
from itertools import combinations

try:
    import msvcrt
    _MSVCRT_AVAILABLE = True
except ImportError:
    _MSVCRT_AVAILABLE = False

import cv2
import numpy as np

# 日本語テキスト描画用（Pillow）
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROBOT_IP = "192.168.0.2"
ROBOT_PORT = 23
ROBOT_TIMEOUT = 10

ARM_RIGHT = 1   # 下に付いている右腕 (画面の右側を担当)
ARM_LEFT = 2    # 上に付いている左腕 (画面の左側を担当)
ARM_NAMES = {ARM_RIGHT: "アーム1(右腕)", ARM_LEFT: "アーム2(左腕)"}

CALIBRATION_FILE = "camera_calibration.json"
BIN_COORDINATES_FILE = "bin_coordinates.json"

# 校正した範囲の外側にどれだけ余裕を見るか (ピクセル)。外れた位置は「届かない可能性あり」と判断する。
DEFAULT_REACH_MARGIN_PX = 40

# ゴミ箱の色検出パラメータ (HSV)
# ※ 赤はHが0度/180度付近の2領域に跨がるため ranges リストで指定
# ※ 照明環境に応じて範囲を調整してください
BIN_COLOR_RANGES = {
    "赤の箱": {
        "ranges": [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255])),
        ],
        "display_bgr": (0, 0, 255),
    },
    "黄の箱": {
        "ranges": [
            (np.array([18, 70, 70]), np.array([40, 255, 255])),
        ],
        "display_bgr": (0, 255, 255),
    },
    "青の箱": {
        "ranges": [
            (np.array([100, 70, 50]), np.array([130, 255, 255])),
        ],
        "display_bgr": (255, 0, 0),
    },
}

# 箱として認識する最小面積 (ピクセル)
MIN_BIN_AREA = 3000

# キャリブレーションの手順 (名前, 使うアーム, 画面上の位置の説明)
#   上→下の順に打つと、4点が長方形に近くなり計算が安定する
CALIBRATION_STEPS = [
    ("右-上", [ARM_RIGHT], "画面の【右側・上寄り】の目印"),
    ("右-下", [ARM_RIGHT], "画面の【右側・下寄り】の目印"),
    ("中央-上", [ARM_RIGHT, ARM_LEFT], "画面の【中央・上寄り】の目印 (両アームを同じ点に合わせる)"),
    ("中央-下", [ARM_RIGHT, ARM_LEFT], "画面の【中央・下寄り】の目印 (両アームを同じ点に合わせる)"),
    ("左-上", [ARM_LEFT], "画面の【左側・上寄り】の目印"),
    ("左-下", [ARM_LEFT], "画面の【左側・下寄り】の目印"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# キー入力 (OpenCVウィンドウ + ターミナル / 全角対応)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY_NONE = 255
KEY_ENTER = 13
KEY_SPACE = 32

# 全角 -> 半角 の対応 (IMEが全角のままでも反応させる)
_ZENKAKU_MAP = {"ｓ": "s", "ｒ": "r", "ｑ": "q", "ｙ": "y", "ｎ": "n", "　": " "}


def get_camera_key(wait_ms=30):
    """
    OpenCVウィンドウとターミナル両方からキー入力を受け付ける。
    WindowsのIMEが干渉して半角入力できない問題を回避するため、
    全角文字（ｓ、ｒ、ｑ）も受け付ける。
    戻り値: 's','r','q','y','n' の ord / 13(Enter) / 32(Space) / 255(押されていない)
    """
    key = cv2.waitKey(wait_ms) & 0xFF

    # ターミナル側でキーが押されている場合 (msvcrt)
    if _MSVCRT_AVAILABLE and msvcrt.kbhit():
        try:
            char = msvcrt.getwch().lower()
            char = _ZENKAKU_MAP.get(char, char)
            if char in ("s", "r", "q", "y", "n"):
                return ord(char)
            if char == "\x1b":
                return ord("q")
            if char in ("\r", "\n"):
                return KEY_ENTER
            if char == " ":
                return KEY_SPACE
        except Exception:
            pass

    # OpenCV ウィンドウ側でのキー判定
    if key in (ord("s"), ord("S")):
        return ord("s")
    if key in (ord("r"), ord("R")):
        return ord("r")
    if key in (ord("q"), ord("Q"), 27):
        return ord("q")
    if key in (ord("y"), ord("Y")):
        return ord("y")
    if key in (ord("n"), ord("N")):
        return ord("n")
    if key in (13, 10):
        return KEY_ENTER
    return key


def ask_yes_no(prompt):
    """ターミナルで y/n を聞く (全角の ｙ / はい でも OK)"""
    ans = input(prompt).strip().lower()
    return ans in ("y", "ｙ", "yes", "はい")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 日本語テキスト描画
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_FONT_CACHE = {}


def get_jp_font(font_size=18):
    """Windowsの日本語フォントを取得（キャッシュ付き）"""
    if not _PIL_AVAILABLE:
        return None
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]
    font_paths = [
        "C:\\Windows\\Fonts\\meiryo.ttc",
        "C:\\Windows\\Fonts\\msgothic.ttc",
        "C:\\Windows\\Fonts\\YuGothM.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                _FONT_CACHE[font_size] = font
                return font
            except Exception:
                continue
    try:
        font = ImageFont.load_default()
        _FONT_CACHE[font_size] = font
        return font
    except Exception:
        return None


def draw_text_jp(img, text, pos, font_size=18, color=(0, 255, 0)):
    """
    OpenCV画像(BGR)に日本語テキストを描画する。
    pos: (x, y) 左上座標
    font_size: フォントサイズ (pt)
    color: BGR形式のタプル
    """
    font = get_jp_font(font_size)
    if font is None:
        cv2.putText(img, text, (pos[0], pos[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2)
        return img
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, (pos[0], pos[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2)
    return img


def draw_split_line(img, calib):
    """左右の担当境界線とラベルを描画する"""
    u = int(calib.split_u)
    h = img.shape[0]
    cv2.line(img, (u, 0), (u, h), (200, 200, 200), 1)
    draw_text_jp(img, "← 左腕(2)", (max(u - 110, 0), h - 28), font_size=16, color=(255, 200, 0))
    draw_text_jp(img, "右腕(1) →", (u + 8, h - 28), font_size=16, color=(0, 200, 255))
    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# カメラ起動
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_CAMERA_INDEX = 1
LAST_CAMERA_INDEX = None   # 直前に入力されたカメラ番号 (同じ実行の中で使い回す)


def ask_camera_index(default=DEFAULT_CAMERA_INDEX):
    """カメラ番号を聞く (カメラを開く直前に呼ぶ)。Enterだけなら default を使う。"""
    global LAST_CAMERA_INDEX
    while True:
        raw = input(f"カメラ番号を入力 (分からなければ {default} のままEnter): ").strip()
        # 全角数字も受け付ける
        raw = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        if raw == "":
            LAST_CAMERA_INDEX = default
            return default
        try:
            LAST_CAMERA_INDEX = int(raw)
            return LAST_CAMERA_INDEX
        except ValueError:
            print("  [WARN] 整数で入力してください (例: 0, 1, 2)")


def open_camera(index, warmup_frames=30, tries=50):
    """
    カメラを開き、実際にフレームが取れることを確認して返す。
    Windowsでは標準方式(MSMF)だと外付けカメラでエラー(-1072873822)が出やすいため、
    DSHOW 方式を先に試し、ダメなら標準方式で開き直す。
    """
    backends = [("DSHOW", getattr(cv2, "CAP_DSHOW", None)), ("標準", None)]
    for name, backend in backends:
        if name == "DSHOW" and backend is None:
            continue
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(tries):
            ret, _frame = cap.read()
            if ret:
                for _ in range(warmup_frames):  # 露出が安定するまで読み飛ばす
                    cap.read()
                print(f"[INFO] カメラ(index={index})を開きました ({name}方式)")
                return cap
            time.sleep(0.1)
        cap.release()
    raise RuntimeError(
        f"カメラ(index={index})から映像を取得できません。"
        "他のプログラムがカメラを使用中でないか、カメラ番号を確認してください。"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# duAro 通信クライアント (WHERE で座標を自動取得)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_where(text):
    """
    `WHERE n:` の応答から (X, Y, Z, O, A, T) を取り出す。

    応答の例 (マニュアル 5-53):
           JT1  JT2  JT3  JT4  JT5  JT6
         -135.0 120.0 90.0 15.0 0.0  0.0
          X[mm]  Y[mm]  Z[mm]  O[deg]  A[deg]  T[deg]
         364.557 155.960 274.000 0.000 0.000 0.000
    「X[mm]」を含む行の次の数値行から先頭6個を読む (軸数が多くても先頭6個でよい)。
    """
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    for i, line in enumerate(lines):
        if "X[mm]" in line:
            for j in range(i + 1, min(i + 4, len(lines))):
                nums = re.findall(r"-?\d+(?:\.\d+)?", lines[j])
                if len(nums) >= 6:
                    return tuple(float(n) for n in nums[:6])
    raise ValueError(f"WHERE の応答から座標を読み取れませんでした:\n{text}")


class DuAroClient:
    """duAro への Telnet 接続。基本の送受信と WHERE による現在位置の取得を行う。"""

    def __init__(self, ip=ROBOT_IP, port=ROBOT_PORT, timeout=ROBOT_TIMEOUT):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.sock = None

    # --- 接続 ---
    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.ip, self.port))
        time.sleep(0.5)
        resp = s.recv(2048).decode("shift_jis", errors="ignore")
        if "login:" in resp.lower():
            s.sendall(b"as\r\n")
            time.sleep(0.5)
            resp = s.recv(2048).decode("shift_jis", errors="ignore")
            if "password:" in resp.lower():
                s.sendall(b"\r\n")
                time.sleep(0.5)
                s.recv(2048)
        self.sock = s
        return self

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    # --- 送受信 ---
    def flush(self):
        """受信バッファに残った古いデータを捨てる"""
        self.sock.settimeout(0.001)
        try:
            while self.sock.recv(4096):
                pass
        except Exception:
            pass

    def send_and_recv(self, command, wait=0.3):
        """1行送信し、wait秒待ってから届いている応答をまとめて返す"""
        self.flush()
        self.sock.settimeout(2.0)
        self.sock.sendall((command + "\r\n").encode("shift_jis"))
        time.sleep(wait)
        self.sock.settimeout(0.3)
        resp = ""
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                resp += chunk.decode("shift_jis", errors="ignore")
        except socket.timeout:
            pass
        return resp

    # --- 現在位置の取得 ---
    def where(self, arm=ARM_RIGHT):
        """アーム arm の現在位置 (X, Y, Z, O, A, T) を返す。単位は mm / deg。"""
        text = self.send_and_recv(f"WHERE {arm}:", wait=0.5)
        pose = parse_where(text)
        # 連続表示モードになっていた場合に備えて空Enterで終了させる
        self.send_and_recv("", wait=0.1)
        return pose


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# キャリブレーション結果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _apply_homography(H, u, v):
    vec = H @ np.array([u, v, 1.0], dtype=np.float64)
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def pixel_to_robot_xy(px, py, homography_matrix):
    """ピクセル座標 (px, py) を、指定したホモグラフィ行列でロボット座標 (X, Y) に変換する"""
    return _apply_homography(np.asarray(homography_matrix, dtype=np.float64), px, py)


class Calibration:
    """H1(アーム1用)・H2(アーム2用)・左右の境界・校正範囲・WHEREで測ったZ をまとめたもの"""

    def __init__(self, H1, H2, split_u, ranges=None, measured_z=None):
        self.H = {ARM_RIGHT: np.array(H1, dtype=np.float64),
                  ARM_LEFT: np.array(H2, dtype=np.float64)}
        self.split_u = float(split_u)
        self.ranges = ranges or {}
        self.measured_z = measured_z or {}   # {"arm1": 平均Z, "arm2": 平均Z}

    def arm_for_pixel(self, u):
        """画面の右側 -> アーム1(右腕) / 左側 -> アーム2(左腕)"""
        return ARM_RIGHT if u >= self.split_u else ARM_LEFT

    def pixel_to_robot(self, u, v, arm):
        """ピクセル(u, v)を、そのアームの座標系の (X, Y) [mm] に変換する"""
        return _apply_homography(self.H[arm], u, v)

    def in_range(self, arm, u, v, margin=DEFAULT_REACH_MARGIN_PX):
        """
        そのアームの校正範囲(+余裕)の中にあるか。
        ホモグラフィは校正した範囲の外では誤差が大きくなる(外挿)ため、
        範囲外は「届くか不明」として扱う。
        """
        box = self.ranges.get(f"arm{arm}")
        if not box:
            return True
        umin, vmin, umax, vmax = box
        return (umin - margin <= u <= umax + margin) and (vmin - margin <= v <= vmax + margin)

    def to_dict(self):
        return {
            "H1_arm1": self.H[ARM_RIGHT].tolist(),
            "H2_arm2": self.H[ARM_LEFT].tolist(),
            "split_u": self.split_u,
            "ranges": self.ranges,
            "measured_z": self.measured_z,
        }

    def save(self, path=CALIBRATION_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] キャリブレーション結果を {path} に保存しました")


def load_calibration(path=CALIBRATION_FILE):
    """camera_calibration.json を読み込んで Calibration を返す"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] '{path}' が見つかりません。先に camera.py の Mode 1 でキャリブレーションを実行してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "H1_arm1" not in d or "H2_arm2" not in d:
        raise ValueError(
            f"[ERROR] '{path}' は古い形式 (片腕・四隅手打ち) のキャリブレーションです。"
            "camera.py の Mode 1 (双腕6点キャリブレーション) をやり直してください。"
        )
    return Calibration(d["H1_arm1"], d["H2_arm2"], d["split_u"], d.get("ranges"), d.get("measured_z"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 双腕 6点キャリブレーション
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _has_collinear_triple(points, min_twice_area=100.0):
    """4点のうち、ほぼ一直線上に並ぶ3点があるか (あるとホモグラフィが不安定になる)"""
    for (x1, y1), (x2, y2), (x3, y3) in combinations(points, 3):
        if abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) < min_twice_area:
            return True
    return False


def compute_homography(pixel_points, robot_points):
    """4組の対応点から、ピクセル -> ロボットXY のホモグラフィ行列を求める"""
    if len(pixel_points) != 4 or len(robot_points) != 4:
        raise ValueError("対応点は4組必要です")
    if _has_collinear_triple(pixel_points) or _has_collinear_triple(robot_points):
        raise ValueError(
            "4点のうち3点がほぼ一直線上に並んでいます。目印を長方形の角のように離して選び直してください"
        )
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(robot_points, dtype=np.float32)
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None:
        raise RuntimeError("ホモグラフィ行列の計算に失敗しました")
    return matrix


def build_calibration(step_pixels, step_poses):
    """
    クリックしたピクセルと WHERE の値から Calibration を組み立てる (GUI/通信に依存しない部分)。

    step_pixels: {ステップ名: (u, v)}
    step_poses : {ステップ名: {アーム番号: (x, y, z)}}
    """
    pix = {ARM_RIGHT: [], ARM_LEFT: []}
    rob = {ARM_RIGHT: [], ARM_LEFT: []}
    zs = {ARM_RIGHT: [], ARM_LEFT: []}
    for name, arms, _desc in CALIBRATION_STEPS:
        for arm in arms:
            x, y, z = step_poses[name][arm]
            pix[arm].append(step_pixels[name])
            rob[arm].append((x, y))
            zs[arm].append(z)

    H1 = compute_homography(pix[ARM_RIGHT], rob[ARM_RIGHT])
    H2 = compute_homography(pix[ARM_LEFT], rob[ARM_LEFT])

    # 左右の境界 = 中央の目印2点のピクセルXの平均
    center_us = [step_pixels[n][0] for n, arms, _ in CALIBRATION_STEPS if len(arms) == 2]
    split_u = int(round(sum(center_us) / len(center_us)))

    ranges, measured_z = {}, {}
    for arm in (ARM_RIGHT, ARM_LEFT):
        us = [p[0] for p in pix[arm]]
        vs = [p[1] for p in pix[arm]]
        ranges[f"arm{arm}"] = [min(us), min(vs), max(us), max(vs)]
        measured_z[f"arm{arm}"] = round(float(np.mean(zs[arm])), 2)

    return Calibration(H1, H2, split_u, ranges, measured_z)


def _show_frame_with_overlay(cap, window_name, title_lines, point=None, footer=None, done_points=None):
    """1フレーム読んで、案内文・クリック点を重ねて表示する"""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("カメラからフレームを取得できませんでした")
    for (u, v), label in (done_points or []):
        cv2.circle(frame, (u, v), 5, (180, 180, 180), -1)
        draw_text_jp(frame, label, (u + 6, max(v - 18, 0)), font_size=14, color=(180, 180, 180))
    for i, line in enumerate(title_lines):
        draw_text_jp(frame, line, (10, 10 + i * 24), font_size=18, color=(0, 255, 0))
    if point:
        cv2.circle(frame, point, 7, (0, 0, 255), -1)
    if footer:
        draw_text_jp(frame, footer, (10, frame.shape[0] - 30), font_size=18, color=(0, 0, 255))
    cv2.imshow(window_name, frame)


def _click_point_on_camera(cap, window_name, title_lines, done_points=None):
    """
    映像上で目印を1回クリックさせ、(u, v) を返す。
    Enter/Space/s で確定、r でやり直し、q で中断。(ウィンドウ・ターミナルどちらでも可 / 全角OK)
    """
    state = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        footer = "Enter/Space/s で確定 / r でやり直し" if state["pt"] else None
        _show_frame_with_overlay(cap, window_name, title_lines, state["pt"], footer, done_points)
        key = get_camera_key(wait_ms=30)
        if key == ord("q"):
            raise KeyboardInterrupt("ユーザーによる中断")
        if key == ord("r"):
            state["pt"] = None
        if key in (KEY_ENTER, KEY_SPACE, ord("s")) and state["pt"]:
            return state["pt"]


def _wait_enter_on_camera(cap, window_name, title_lines, point, done_points=None):
    """
    映像を流したまま Enter を待つ (ウィンドウ・ターミナルどちらでも可)。
    input() と違い、待っている間も映像が止まらない。q で中断。
    """
    cv2.setMouseCallback(window_name, lambda *a: None)
    while True:
        _show_frame_with_overlay(cap, window_name, title_lines, point,
                                 "合わせたら Enter (q で中断)", done_points)
        key = get_camera_key(wait_ms=30)
        if key == ord("q"):
            raise KeyboardInterrupt("ユーザーによる中断")
        if key in (KEY_ENTER, KEY_SPACE, ord("s")):
            return


def run_homography_calibration(camera_index, client):
    """
    Mode 1: 双腕 6点キャリブレーション (WHERE で座標を自動取得)
    camera_index が None のときは、カメラを開く直前にカメラ番号を聞く。
    """
    print("\n[INFO] 机の上に、目印(シールなど)を6か所貼ってください:")
    print("       右側2か所 / 中央2か所 / 左側2か所 (それぞれ 上と下 に離す)")
    print("       中央の2か所は、アーム1(右腕)・アーム2(左腕) の両方が届く位置にしてください")
    print("       ※ 中央の2か所の位置が、左右の担当の境界線になります")

    if camera_index is None:
        camera_index = ask_camera_index()
    cap = open_camera(camera_index)
    window = "calibration (click mark, Enter: OK, r: redo, q: quit)"
    cv2.namedWindow(window)

    step_pixels, step_poses = {}, {}
    done_points = []
    try:
        for n, (name, arms, desc) in enumerate(CALIBRATION_STEPS, 1):
            arm_text = " と ".join(ARM_NAMES[a] for a in arms)
            head = f"ステップ {n}/{len(CALIBRATION_STEPS)}: {name}  ({arm_text})"
            print(f"\n=== {head} ===")
            print(f"  {desc}")
            print("  1) 映像上で、その目印をクリックして Enter で確定")
            pixel = _click_point_on_camera(
                cap, window, [head, desc, "目印の位置をクリックしてください"], done_points,
            )
            step_pixels[name] = pixel
            print(f"  クリック位置: {pixel}")

            step_poses[name] = {}
            for arm in arms:
                msg = f"{ARM_NAMES[arm]} の先端を、その目印の真上に合わせて Enter"
                print(f"  2) {msg}  (ウィンドウ・ターミナルどちらで押してもOK)")
                _wait_enter_on_camera(cap, window, [head, msg], pixel, done_points)
                x, y, z, o, a, t = client.where(arm)
                step_poses[name][arm] = (x, y, z)
                print(f"     {ARM_NAMES[arm]}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}  を取得しました")
            done_points.append((pixel, name))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    calib = build_calibration(step_pixels, step_poses)
    print(f"\n[INFO] 左右の境界 split_u = {calib.split_u:.0f} px  (右 -> アーム1(右腕) / 左 -> アーム2(左腕))")
    print(f"[INFO] H1 (アーム1):\n{calib.H[ARM_RIGHT]}\n[INFO] H2 (アーム2):\n{calib.H[ARM_LEFT]}")
    z1, z2 = calib.measured_z.get("arm1"), calib.measured_z.get("arm2")
    print(f"[INFO] 目印に合わせた時の先端Z (平均): アーム1={z1:.1f} mm / アーム2={z2:.1f} mm")
    if abs(z1 - z2) > 10.0:
        print("[WARN] アーム1とアーム2で Z が 10mm 以上違います。")
        print("       duaro_waste_sorting.py の ARM_Z (掴む高さ・机面限界) をアームごとに見直してください。")
    calib.save()
    return calib


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ゴミ箱の色検出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_bins_in_frame(frame):
    """
    1フレームから赤・黄・青の箱を HSV 色検出で見つけ、各箱の中心ピクセル座標を返す。

    戻り値:
        centers      : {"赤の箱": (cx, cy) or None, ...}
        contours_map : {"赤の箱": 輪郭 or None, ...}  (描画用)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    kernel = np.ones((7, 7), np.uint8)

    centers, contours_map = {}, {}
    for bin_name, cfg in BIN_COLOR_RANGES.items():
        mask = None
        for lower, upper in cfg["ranges"]:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_BIN_AREA and area > best_area:
                best, best_area = cnt, area

        centers[bin_name] = None
        contours_map[bin_name] = None
        if best is not None:
            M = cv2.moments(best)
            if M["m00"] > 0:
                centers[bin_name] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                contours_map[bin_name] = best
    return centers, contours_map


def make_bin_entry(pixel, calib):
    """
    箱の保存用データ: ピクセル位置と、各アームから見たロボット座標。
    箱は中央に置く前提なので、どちらのアームからも使えるよう両方の座標を持つ。
    """
    u, v = pixel
    entry = {"pixel": [int(u), int(v)]}
    for arm in (ARM_RIGHT, ARM_LEFT):
        x, y = calib.pixel_to_robot(u, v, arm)
        entry[f"arm{arm}"] = {"x": round(x, 2), "y": round(y, 2),
                              "in_range": bool(calib.in_range(arm, u, v))}
    return entry


def detect_bins_from_camera(camera_index, calib):
    """
    カメラ映像でゴミ箱を検出し、確認しながら保存用データを作る。
    's'/Enter/Space: 確定  'r': やり直し  'q': 中断  (ウィンドウ・ターミナルどちらでも / 全角OK)
    戻り値: {"赤の箱": {"pixel": [u, v], "arm1": {...}, "arm2": {...}}, ...}
    """
    if camera_index is None:
        camera_index = ask_camera_index()
    cap = open_camera(camera_index)
    window = "Bin Detection (s: save, r: retry, q: quit)"
    cv2.namedWindow(window)
    print("\n[INFO] 赤・黄・青の箱がカメラに映るように配置してください (箱は中央に置く)")
    print("[INFO] 's' で確定保存 / 'r' で再検出 / 'q' で中断")

    result = {}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("カメラからフレームを取得できませんでした")
            centers, contours_map = detect_bins_in_frame(frame)
            display = frame.copy()
            draw_split_line(display, calib)

            y = 30
            all_found = True
            for name, cfg in BIN_COLOR_RANGES.items():
                color = cfg["display_bgr"]
                c, cnt = centers.get(name), contours_map.get(name)
                if c is not None and cnt is not None:
                    cv2.drawContours(display, [cnt], -1, color, 2)
                    cv2.circle(display, c, 8, color, -1)
                    reach = "/".join(
                        f"{a}{'OK' if calib.in_range(a, c[0], c[1]) else '範囲外?'}"
                        for a in (ARM_RIGHT, ARM_LEFT)
                    )
                    draw_text_jp(display, f"OK {name}: px({c[0]},{c[1]})  アーム{reach}",
                                 (10, y), font_size=18, color=color)
                else:
                    draw_text_jp(display, f"NG {name}: 未検出", (10, y), font_size=18, color=(0, 0, 255))
                    all_found = False
                y += 28
            if all_found:
                cv2.putText(display, "ALL BINS DETECTED - Press 's' to save", (10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window, display)
            key = get_camera_key(wait_ms=30)
            if key == ord("q"):
                raise KeyboardInterrupt("ユーザーによる中断")
            elif key == ord("r"):
                print("[INFO] 再検出中...（箱の位置や向きを調整してください）")
                continue
            elif key in (ord("s"), KEY_ENTER, KEY_SPACE):
                result = {n: make_bin_entry(c, calib) for n, c in centers.items() if c is not None}
                missing = [n for n, c in centers.items() if c is None]
                if missing:
                    print(f"[WARN] 以下の箱が検出できていません: {', '.join(missing)}")
                    if not ask_yes_no("  検出できた箱だけで保存しますか？ (y/n): "):
                        print("[INFO] 再検出します。箱を映し直してください")
                        continue
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return result


def save_bin_coordinates(bin_coords, path=BIN_COORDINATES_FILE):
    """ゴミ箱の座標を JSON に保存する"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bin_coords, f, ensure_ascii=False, indent=2)
    print(f"[INFO] ゴミ箱座標を {path} に保存しました")


def load_bin_coordinates(path=BIN_COORDINATES_FILE):
    """ゴミ箱の座標を JSON から読み込む"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] '{path}' が見つかりません。先に camera.py の Mode 2 でゴミ箱検出を実行してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_bin_detection(camera_index=None):
    """Mode 2: ゴミ箱の検出と保存 (camera_index が None ならカメラを開く直前に聞く)"""
    calib = load_calibration()
    bins = detect_bins_from_camera(camera_index, calib)
    if not bins:
        print("[WARN] ゴミ箱が1つも検出されませんでした")
        return
    print("\n[INFO] 検出されたゴミ箱:")
    for name, e in bins.items():
        print(f"  {name}: pixel={e['pixel']}  "
              f"アーム1=({e['arm1']['x']:.1f}, {e['arm1']['y']:.1f}) "
              f"アーム2=({e['arm2']['x']:.1f}, {e['arm2']['y']:.1f})")
    save_bin_coordinates(bins)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン（対話メニュー）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _calibrate():
    client = DuAroClient()
    print(f"[INFO] duAro ({client.ip}:{client.port}) に接続中...")
    client.connect()
    try:
        # カメラ番号は、キャリブレーションでカメラを開く直前に聞かれる
        return run_homography_calibration(None, client)
    finally:
        client.close()


if __name__ == "__main__":
    print("=" * 50)
    print("  duAro カメラセットアップ (双腕)")
    print("=" * 50)

    print("\n  1. 双腕6点キャリブレーション (WHERE で座標を自動取得)")
    print("  2. ゴミ箱の検出 (赤・黄・青)")
    print("  3. 両方実行 (1 -> 2)")
    print("  q. 終了")
    choice = input("\n選択: ").strip().translate(str.maketrans("１２３ｑ", "123q")).lower()

    try:
        if choice == "1":
            _calibrate()
        elif choice == "2":
            run_bin_detection()          # カメラを開く直前にカメラ番号を聞く
        elif choice == "3":
            _calibrate()
            print("\n--- 続いてゴミ箱の検出に進みます (カメラ番号はさっき入力したものを使います) ---\n")
            run_bin_detection(LAST_CAMERA_INDEX)
        elif choice == "q":
            print("[INFO] 終了")
        else:
            print("[ERROR] 無効な選択です")
    except KeyboardInterrupt:
        print("\n[INFO] 中断されました")
    except Exception as e:
        print(f"[ERROR] {e}")

    print("\n[INFO] カメラセットアップ完了")