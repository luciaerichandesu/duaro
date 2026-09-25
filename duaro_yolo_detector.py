"""
duaro_yolo_detector.py (ファイル1.5: ゴミ分別対応 YOLO26 物体検出モジュール)

・YOLO26で検出した物体を「紙ごみ」「プラスチック」「缶ビン」に自動分類します。
・duAroが掴むための中心ピクセル座標 (cx, cy) と、分別カテゴリを返します。

単体テスト:
    python duaro_yolo_detector.py
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 分別設定（マッピングと色）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分別カテゴリの定義
CATEGORY_PAPER = "紙ごみ"
CATEGORY_PLASTIC = "プラスチック"
CATEGORY_CAN_BOTTLE = "缶ビン"

# YOLO標準(COCO)の検出クラスをどのゴミに分類するかの対応表
# ※ 独自のカスタム学習モデルを使う場合は、そのクラス名をキーに指定してください
CLASS_TO_CATEGORY = {
    # 紙ごみ
    "book": CATEGORY_PAPER,
    # 紙ごみ(紙コップを使うがcocoデータセットではコップという括りで紙かプラか判別できないのでここでもう指定する)
    "cup": CATEGORY_PAPER,          # プラスチックコップなど
    #プラスチック
    "toothbrush": CATEGORY_PLASTIC,
    "mouse": CATEGORY_PLASTIC,
    # 缶・ビン
    "bottle": CATEGORY_CAN_BOTTLE,    # ペットボトル・ビン・缶など
    "wine glass": CATEGORY_CAN_BOTTLE,
}

# 画面表示用の枠色 (BGR)#物体を枠線で囲むときの色
CATEGORY_COLORS = {
    CATEGORY_PAPER: (255, 191, 0),       # 紙ごみは水色
    CATEGORY_PLASTIC: (0, 215, 255),     # プラスチックは黄色
    CATEGORY_CAN_BOTTLE: (0, 0, 255),    # 缶・ビンは赤色
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 検出＆分別クラス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class WasteDetector:
    def __init__(self, model_path="yolo26x.pt", conf_thresh=0.25):
        print(f"[INFO] YOLO26モデルをロード中: {model_path}")
        self.model = YOLO(model_path)#モデルにYOLOを準備
        self.conf_thresh = conf_thresh#閾値を後でも使えるように保存する

    def detect(self, frame):#frame、カメラ画像を受け取る
        """
        カメラ映像から物体を検出し、ゴミの分別カテゴリと中心座標を返す。

        戻り値:
            best_target: {
                "pixel": (cx, cy),       # 掴む位置 (ピクセル)
                "category": 分別カテゴリ ("紙ごみ" / "プラスチック" / "缶ビン"),
                "class": 認識されたクラス名,
                "conf": 信頼度,
                "box": (x1, y1, x2, y2)
            } ※対象が見つからない場合は None
            display_frame: 画面表示用画像
        """
        #推論
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,#閾値の処理
            verbose=False#ターミナル画面への詳細なログ出力をオフ
        )

        display_frame = frame.copy()
        best_target = None#一番つかみたいゴミをいれておく空の箱
        highest_conf = -1.0#信頼度の初期値(自信の度合い)、1.0~0だがはじめは-1

        #何も見つからなかったら0ですぐに処理修了
        if not results or len(results[0].boxes) == 0:
            return None, display_frame

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]#ゴミの名前
            conf = float(box.conf[0].item())#自信度、信頼度

            # 分別対応表に登録されているかチェック
            category = CLASS_TO_CATEGORY.get(cls_name.lower())
            if category is None:
                # 分別対象外の物体（人間、机、椅子など）は無視する
                continue

            # バウンディングボックス [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 物体の中心座標 (cx, cy)　真ん中を狙うので足して2でわる
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 最も信頼度が高いものをターゲットにする
            if conf > highest_conf:
                highest_conf = conf
                best_target = {
                    "pixel": (cx, cy),
                    "category": category,
                    "class": cls_name,
                    "conf": conf,#自信度
                    "box": (x1, y1, x2, y2)#バウンディングボックスの角の座標
                }

            # 画面枠の描画（カテゴリごとの色）
            box_color = CATEGORY_COLORS.get(category, (0, 255, 0))#もし紙、プラ、缶ビン以外のものがあれば緑で囲む
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)#ここで囲むのを実行する

            # ラベル表示
            label_text = f"[{category}] {cls_name} ({conf:.2f})"#画面上にそれぞれ分類、クラス名、自信度を表示する
            cv2.putText(
                display_frame, label_text, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2
            )

        # 1位に選ばれたターゲットの中心に目印を描画
        if best_target:
            tcx, tcy = best_target["pixel"]#中心座標取り出し
            cat = best_target["category"]#分類名を取り出し
            color = CATEGORY_COLORS.get(cat, (0, 0, 255))

            #1位のゴミのど真ん中に、丸を描く
            cv2.circle(display_frame, (tcx, tcy), 6, color, -1)#6は〇の半径、-1で塗りつぶす

            #丸のすぐ右斜め上に大きい文字でTARGETとかく
            cv2.putText(
                display_frame, f"TARGET -> {cat}", (tcx + 8, tcy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return best_target, display_frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 単体テスト（カメラだけで確認）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    detector = WasteDetector()#先ほど定義したclassがうごく

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] カメラを開けませんでした")
        raise SystemExit(1)

    print("\n[INFO] ゴミ分別 検出テスト開始 ('q' キーで終了)")
    print("【分別色】水色: 紙ごみ / 黄色: プラスチック / 赤色: 缶ビン\n")

    while True:#breakがでるまで無限に繰り返す
        ret, frame = cap.read()#cap.read()は写真撮影の意味、retに撮影成功したかのTrue,Falseが入っており、frameにはその画像が入ってる
        if not ret:
            break

        #targetがロボット用のデータ(座標やクラス名など)、veiwがただの画像
        target, view = detector.detect(frame)#先ほど撮った画像をyolo26にわたし推論

        if target:
            cat = target["category"]
            cx, cy = target["pixel"]
            conf = target["conf"]
            #写真の上に枠線や文字を書きこむ
            cv2.putText(
                view, f"Detected: {cat} ({cx}, {cy}) conf:{conf:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        #書き終わったものをpcの画面上にだす
        cv2.imshow("Waste Sorting Detector Test", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()#カメラの電源をきる
    cv2.destroyAllWindows()#開いていた画面のすべてのwindowを閉じる
