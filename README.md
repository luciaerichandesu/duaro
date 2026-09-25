# duAro ゴミ自動分別システム

YOLO26 による物体検出と川崎重工 duAro（双腕ロボット）を組み合わせて、  
机の上のゴミを自動で **認識 → 分類 → 対応する箱へ投入** するシステムです。

---

## 目次

0. [成果物の作成方法](vibe codingについて)
1. [ファイル構成](#ファイル構成)
2. [ファイルの連携（importの仕組み）](#ファイルの連携importの仕組み)
3. [必要なもの](#必要なもの)
4. [セットアップ手順](#セットアップ手順)
5. [使い方](#使い方)
6. [キー操作一覧](#キー操作一覧)
7. [設定パラメータ](#設定パラメータ)
8. [YOLO分別対応表](#yolo分別対応表)
9. [トラブルシューティング](#トラブルシューティング)

---

## 成果物作成手法について
このプロジェクトは、googleのantigravtyを活用した「vibe coding」によって構築された実験的プロジェクト及び高専のPBL課題です。私がほとんどコードを書かない上に高速性を優先して全てのコードを確認できていないで、コードの乱れや予期せぬ挙動があるかもしれませんが、温かい目で見ていただけると幸いです。

---

## ファイル構成

```
duaro/
├── camera.py                 … カメラキャリブレーション＆ゴミ箱色検出
├── duaro_yolo_detector.py    … YOLO26 で物体を検出し、ゴミのカテゴリを判定
├── air.py                    … 空気ポンプ制御（ハンドの開閉）
├── duaro_waste_sorting.py    … メインプログラム（↑3つを統合して動かす）
├── install.txt               … pip install コマンドのメモ
└── README.md                 … このファイル
```

| ファイル | 種類 | 単体実行 | 説明 |
|---|---|---|---|
| `camera.py` | モジュール | `python camera.py` | カメラで机の四隅をクリック → ホモグラフィ行列を計算して `camera_calibration.json` に保存。ゴミ箱の色検出機能も持つ |
| `duaro_yolo_detector.py` | モジュール | `python duaro_yolo_detector.py` | YOLO26でカメラ映像から物体を検出し、紙ごみ/プラスチック/缶ビンに分類。単体実行でカメラテスト可能 |
| `air.py` | モジュール | `python air.py` | duAro に登録した `pump_on` / `pump_off` プログラムを実行してハンドを開閉。単体実行でポンプテスト可能 |
| `duaro_waste_sorting.py` | **メイン** | `python duaro_waste_sorting.py` | 上記3モジュールを統合し、ゴミ検出→ピッキング→分別投入を制御するメインプログラム |
| `install.txt` | ドキュメント | — | `pip install` コマンドのメモ。Python から読み込まれることはない |

### 自動生成されるファイル

| ファイル | 生成元 | 内容 |
|---|---|---|
| `camera_calibration.json` | `camera.py` Mode 1 | ホモグラフィ行列（ピクセル座標→ロボット座標の変換データ）と机のZ座標 |
| `bin_coordinates.json` | `duaro_waste_sorting.py` 起動時 | ゴミ箱（白・黄・黒）のロボット座標 |

---

## ファイルの連携（importの仕組み）

`python duaro_waste_sorting.py` を実行すると、冒頭の `import` 文で他の3ファイルが自動的に読み込まれます。  
自分で個別に実行する必要はありません。

```
duaro_waste_sorting.py（メインプログラム）
│
├── from camera import ...               ← camera.py の関数・定数を使う
│     ├── load_calibration()                 JSONからキャリブレーション読み込み
│     ├── pixel_to_robot_xy()                ピクセル座標→ロボット座標に変換
│     ├── detect_bins_in_frame()             ゴミ箱を色で検出
│     ├── save_bin_coordinates()             ゴミ箱座標をJSONに保存
│     ├── load_bin_coordinates()             ゴミ箱座標をJSONから読み込み
│     └── BIN_COORDINATES_FILE               ファイルパスの定数
│
├── from duaro_yolo_detector import ...  ← duaro_yolo_detector.py のクラス・定数を使う
│     ├── WasteDetector                      YOLO検出クラス（モデルを保持して何度も使う）
│     ├── CATEGORY_PAPER                     "紙ごみ" の定数
│     ├── CATEGORY_PLASTIC                   "プラスチック" の定数
│     └── CATEGORY_CAN_BOTTLE                "缶ビン" の定数
│
└── from air import ...                  ← air.py の関数を使う
      ├── pump_on()                          ポンプON（ハンドを閉じて掴む）
      └── pump_off()                         ポンプOFF（ハンドを開いて離す）
```

---

## 必要なもの

### ハードウェア

| 機器 | 備考 |
|---|---|
| 川崎重工 duAro（双腕ロボット） | IPアドレス: `192.168.0.2`（初期設定） |
| USB外部カメラ | カメラインデックス: `1`（外部カメラ） |
| PC（Windows） | duAro とLANケーブルで接続 |
| ゴミ箱 3つ | 白色・黄色・黒色の箱 |
| 空気ポンプ式ハンド | duAro のクランプ1に接続 |

### ソフトウェア

| 名前 | 用途 |
|---|---|
| Python 3.8 以上 | プログラムの実行環境 |
| ultralytics | YOLO26 による物体検出 |
| opencv-python | カメラ映像の取得・画像処理 |
| numpy | 座標計算などの数値計算 |

---

## セットアップ手順

### STEP 0: ライブラリのインストール（初回のみ）

コマンドプロンプトを開いて以下を実行：

```bash
cd C:\Users\tokio\Downloads\duaro
pip install ultralytics opencv-python numpy
```

### STEP 1: duAro 側の準備（初回のみ）

krtermまたはティーチペンダントで、以下の **3つのASプログラム** を登録します：

```
① アーム移動用
.PROGRAM goto1()
  JMOVE p1
.END

② ハンドを閉じる（掴む）
.PROGRAM pump_on()
  OPENI 1
.END

③ ハンドを開く（離す）
.PROGRAM pump_off()
  CLOSEI 1
.END
```

### STEP 2: カメラキャリブレーション（初回 or 机の配置を変えた時）

#### 事前準備

duAro を机の4隅（左上→右上→右下→左下）の位置に手動で動かし、  
krtermで `HERE p_now` を実行してそれぞれの `X, Y` 座標をメモしておきます。

#### 実行

```bash
cd C:\Users\tokio\Downloads\duaro
python camera.py
```

#### 操作

1. カメラ番号を聞かれるのでそのまま Enter（デフォルト: 1）
2. メニューで `1`（ホモグラフィキャリブレーション）を選択
3. カメラ映像が表示される → 机の四隅を「左上→右上→右下→左下」の順にクリック
4. 各隅に対応する duAro 座標（X, Y）をメモした値を入力
5. 机の高さ（Z座標, mm）を入力
6. `camera_calibration.json` が自動生成される

---

## 使い方

### 本番実行

```bash
cd C:\Users\tokio\Downloads\duaro
python duaro_waste_sorting.py
```

### 起動時に自動的に行われること

1. `camera_calibration.json` を読み込み（キャリブレーションデータ）
2. カメラを起動
3. カメラ映像からゴミ箱（白・黄・黒）を色で自動検出 → ロボット座標に変換
4. duAro にネットワーク接続（`192.168.0.2:23`）
5. モータ電源 ON
6. YOLO26 モデルを読み込み（初回は数秒かかる）
7. カメラ映像ウィンドウが開く → **ここからキー操作で制御**

### 全自動分別の流れ（`a` キー）

```
'a' キーを押す
  ↓
カメラでゴミを検出（例: 紙コップ → 紙ごみ）
  ↓
アームが紙コップの上空へ移動
  ↓
下降 → ポンプON（掴む）→ 上昇
  ↓
対応する箱（白の箱）の上空へ移動
  ↓
下降 → ポンプOFF（離す = 投下）→ 上昇
  ↓
ホームポジションへ戻る
  ↓
次のゴミを探す…（あれば繰り返し）
  ↓
5回連続で何も見つからない →「全完了！」
```

---

## キー操作一覧

### camera.py（キャリブレーション時）

| キー | 動作 |
|---|---|
| 左クリック | 机の隅をクリック（4回） |
| `r` | クリックをリセット |
| `q` | 中断 |

### duaro_waste_sorting.py（本番実行時）

| キー | 動作 |
|---|---|
| `s` | **手動モード** — 画面に映っているゴミを1個だけ分別 |
| `a` | **全自動モード** — ゴミがなくなるまで自動で繰り返す |
| `q` | プログラム終了（全自動モード中は緊急停止） |

### air.py（ポンプ単体テスト時）

| キー | 動作 |
|---|---|
| `p` | ポンプ ON（ハンドを閉じる） |
| `o` | ポンプ OFF（ハンドを開く） |
| `q` | 終了 |

---

## 設定パラメータ

`duaro_waste_sorting.py` の冒頭で以下のパラメータを調整できます。

### duAro 接続設定

| 変数名 | 初期値 | 説明 |
|---|---|---|
| `ROBOT_IP` | `"192.168.0.2"` | duAro の IP アドレス |
| `ROBOT_PORT` | `23` | Telnet ポート番号 |
| `TIMEOUT` | `10` | 接続タイムアウト（秒） |
| `ARM` | `1` | 使用するアーム番号 |
| `POSE_NAME` | `"p1"` | POINT 指令で使う変数名 |
| `PROGRAM_NAME` | `"goto1"` | 移動用の AS プログラム名 |
| `PUMP_ON_PROGRAM` | `"pump_on"` | ハンドを閉じる AS プログラム名 |
| `PUMP_OFF_PROGRAM` | `"pump_off"` | ハンドを開く AS プログラム名 |
| `CAMERA_INDEX` | `1` | カメラ番号（外部カメラ = 1） |

### 高さ設定（単位: mm）

| 変数名 | 初期値 | 説明 |
|---|---|---|
| `Z_APPROACH_OFFSET` | `60.0` | 物体の上空待機高さ（机面 + 60mm） |
| `Z_GRASP_OFFSET` | `15.0` | 掴む高さ（机面 + 15mm、ワークの厚みに合わせて調整） |
| `BIN_Z_DROP` | `180.0` | 箱の中でゴミを離す高さ |
| `BIN_Z_SAFE` | `280.0` | 箱の上空安全高さ |
| `HOME_POS` | `{x:250, y:0, z:300}` | 待機位置（カメラの視野を遮らない位置） |

### 全自動モード設定

| 変数名 | 初期値 | 説明 |
|---|---|---|
| `AUTO_NO_DETECT_LIMIT` | `5` | 連続で何回未検出なら「全完了」と判断するか |
| `AUTO_STABILIZE_WAIT` | `1.0` | 分別後、次の検出前の待機時間（秒） |
| `AUTO_MAX_FAILURES` | `3` | 連続失敗がこの回数に達したら自動停止 |

---

## YOLO分別対応表

`duaro_yolo_detector.py` の `CLASS_TO_CATEGORY` で定義。  
YOLO26（COCO データセット）の検出クラス名をゴミのカテゴリに対応させています。

| YOLO 検出クラス名 | 分別カテゴリ | 投入先 | 枠の色 |
|---|---|---|---|
| `book` | 紙ごみ | 白の箱 | 水色 |
| `cup` | 紙ごみ | 白の箱 | 水色 |
| `toothbrush` | プラスチック | 黄の箱 | 黄色 |
| `mouse` | プラスチック | 黄の箱 | 黄色 |
| `bottle` | 缶ビン | 黒の箱 | 赤色 |
| `wine glass` | 缶ビン | 黒の箱 | 赤色 |

カスタム学習モデルを使う場合は `CLASS_TO_CATEGORY` のキーを自分のモデルのクラス名に変更してください。

---

## トラブルシューティング

### `ModuleNotFoundError: No module named 'cv2'`
ライブラリが未インストール。以下を実行：
```bash
pip install ultralytics opencv-python numpy
```

### `ModuleNotFoundError: No module named 'camera'`
`camera.py` が `duaro_waste_sorting.py` と同じフォルダにない。  
5ファイルすべてが同じ `duaro` フォルダにあることを確認。

### `FileNotFoundError: 'camera_calibration.json' が見つかりません`
キャリブレーション未実施。先に `python camera.py` を実行する。

### `[ERROR] カメラを開けませんでした`
USB カメラが接続されていないか、カメラ番号が違う。  
別のカメラ番号（`0` や `2`）を試す場合は `CAMERA_INDEX` を変更。

### `[ERROR] 接続拒否: 192.168.0.2:23 に接続できません`
duAro の電源が入っていない、またはLAN ケーブルが未接続。  
PC と duAro が同じネットワークにいることを確認。

### `[ERROR] ロボット異常停止`
duAro 側でエラーが発生（動作範囲外など）。  
ティーチペンダントでエラーを解除し、`HOME_POS` や高さ(`Z_*`)の値を見直す。

### ゴミ箱が検出されない
照明条件が合っていない可能性。  
`camera.py` の `BIN_COLOR_RANGES`（HSV の上下限値）を環境に合わせて調整。

### 特定の物体が分別されない
`duaro_yolo_detector.py` の `CLASS_TO_CATEGORY` にそのクラス名が登録されていない。  
YOLO が検出するクラス名（COCO データセット80クラス）を確認して追加する。
