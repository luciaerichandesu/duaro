import re
import socket
import time
 
 
def connect(ip, port, timeout):
    """duAroにTelnet接続し、自動ログインする"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((ip, port))
 
    time.sleep(0.5)
    initial_resp = s.recv(2048).decode("shift_jis", errors="ignore")
 
    if "login:" in initial_resp.lower() or "as" in initial_resp.lower():
        s.sendall(b"as\r\n")
        time.sleep(0.5)
        login_resp = s.recv(2048).decode("shift_jis", errors="ignore")
 
        if "password:" in login_resp.lower():
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
    sock.settimeout(0.0001)
    try:
        while sock.recv(1024):
            pass
    except Exception:
        pass
 
 
def send_and_recv(sock, command, wait=0.3, bufsize=4096):
    try:
        flush_buffer(sock)
        sock.settimeout(2.0)
        sock.sendall((command + "\r\n").encode("shift_jis"))
        time.sleep(wait)
 
        sock.settimeout(1.0)
        resp = ""
        try:
            while True:
                chunk = sock.recv(bufsize).decode("shift_jis", errors="ignore")
                if not chunk:
                    break
                resp += chunk
        except socket.timeout:
            pass
        return resp
    except Exception as e:
        print(f"[ERROR] 送信/受信エラー: {e}")
        return ""
 
 
_ERROR_CODE_PATTERN = re.compile(r"\([EP]\d{4}\)")
 
 
def wait_for_completion(sock, max_wait=60):
    """
    「プログラムが終了しました」を検知するまで待つ。
    (E####)/(P####)のようなエラーコードや、ホールド・途中停止・動作範囲外の
    メッセージを検知した場合は、完了を待たずに即座に失敗として返す。
    """
    sock.settimeout(0.5)
    start = time.time()
    full_resp = ""
    while time.time() - start < max_wait:
        try:
            chunk = sock.recv(2048).decode("shift_jis", errors="ignore")
            if not chunk:
                break
            full_resp += chunk
 
            if "プログラムが終了しました" in full_resp:
                print(f"[INFO] 完了を検知: {full_resp.strip().splitlines()[-1]}")
                return True, full_resp.strip()
 
            if (
                _ERROR_CODE_PATTERN.search(full_resp)
                or "ホールド" in full_resp
                or "途中停止" in full_resp
                or "動作範囲外" in full_resp
            ):
                print(f"[ERROR] duAro側でエラー/停止を検知:\n{full_resp.strip()}")
                return False, full_resp.strip()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[ERROR] 受信エラー: {e}")
            return False, full_resp.strip()
 
    print(f"[WARN] {max_wait}秒以内に完了メッセージを検知できませんでした")
    return False, full_resp.strip()
 
 
def motor_on(sock, arm=1):
    """モータ電源をONにする（ZPOWER指令）"""
    print(f"[INFO] モータ電源ON: ZPOWER {arm}: ON")
    resp = send_and_recv(sock, f"ZPOWER {arm}: ON", wait=1.0)
    print(f"[RECV] {resp.strip()}")
    return resp
 
 
def run_program(sock, program_name, arm=1):
    """
    duAro側にあらかじめ登録済みのASプログラムをEXECUTEで実行する。
    完了待ちは行わず、送信したら即座にreturnする。
    """
    try:
        print(f"[INFO] プログラム実行: EXECUTE {arm}:{program_name}")
        flush_buffer(sock)
        sock.settimeout(2.0)
        sock.sendall(f"EXECUTE {arm}:{program_name}\r\n".encode("shift_jis"))
        return True
    except Exception as e:
        print(f"[ERROR] 送信エラー: {e}")
        return False
 
 
def pump_on(sock, arm=1, program_name="pump_on"):
    """
    空気ポンプ(クランプ1の電磁弁)をONにする。
    duAro側に .PROGRAM pump_on() / OPENI 1 / .END を登録しておく必要がある。
    """
    print("[INFO] ポンプON")
    if run_program(sock, program_name, arm=arm):
        return wait_for_completion(sock)
    return False, ""
 
 
def pump_off(sock, arm=1, program_name="pump_off"):
    """
    空気ポンプ(クランプ1の電磁弁)をOFFにする。
    duAro側に .PROGRAM pump_off() / CLOSEI 1 / .END を登録しておく必要がある。
    """
    print("[INFO] ポンプOFF")
    if run_program(sock, program_name, arm=arm):
        return wait_for_completion(sock)
    return False, ""
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    IP, PORT, TIMEOUT = "192.168.0.2", 23, 10
    ARM = 1
    PUMP_ON_PROGRAM = "pump_on"
    PUMP_OFF_PROGRAM = "pump_off"
 
    print(f"[INFO] {IP}:{PORT} に接続中...")
    try:
        sock = connect(IP, PORT, TIMEOUT)
    except ConnectionRefusedError:
        print(f"[ERROR] 接続拒否: {IP}:{PORT} に接続できません")
        raise SystemExit(1)
    except socket.timeout:
        print(f"[ERROR] タイムアウト: {TIMEOUT}秒以内に応答がありません")
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] 接続エラー: {e}")
        raise SystemExit(1)
    print("[INFO] 接続・ログイン完了")
 
    print("[INFO] duAro側に以下の2つのASプログラムが登録されている必要があります:")
    print("           .PROGRAM pump_on()  / OPENI 1 / .END")
    print("           .PROGRAM pump_off() / CLOSEI 1 / .END")
    print("[INFO] 'p'でポンプON, 'o'でポンプOFF, 'q'で終了")
 
    try:
        motor_on(sock, arm=ARM)
 
        while True:
            print("\n操作を入力 (p=ON, o=OFF, q=終了): ", end="", flush=True)
            cmd = input().strip().lower()
 
            if cmd == "q":
                break
            elif cmd == "p":
                pump_on(sock, arm=ARM, program_name=PUMP_ON_PROGRAM)
            elif cmd == "o":
                pump_off(sock, arm=ARM, program_name=PUMP_OFF_PROGRAM)
            else:
                print("[WARN] p, o, q のいずれかを入力してください")
 
    finally:
        disconnect(sock)
        print("[INFO] 切断しました")