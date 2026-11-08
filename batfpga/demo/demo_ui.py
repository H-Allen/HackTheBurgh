#!/usr/bin/env python3
"""
demo_ui.py - live demo: play bat audio out loud on the laptop while the PYNQ FPGA classifies the
same recording (streamed from the board's SD card), and show its detections on a spectrogram.

    pip install pyserial numpy scipy matplotlib
    python demo/demo_ui.py --port /dev/tty.usbserial-XXXX1      # Mac   (Windows: --port COM5)
    -> open http://localhost:8000

Before starting: power the board, wait for boot, CLOSE screen/PuTTY (one program per serial port).
The script logs in if needed (xilinx/xilinx), becomes root, cd's to --board-dir and starts
`python3 run_bat.py --serve`, which loads the overlay once and then waits for PLAY commands.

The clips in --clips must be the same files as in --board-dir on the SD card (same names).
Two buttons per clip:
  real time  - laptop plays the clip at true speed (bat calls are ultrasonic: silent to us);
               board streams at real time.
  x10 slow   - laptop plays it 10x slowed (time expansion: 20-110 kHz -> 2-11 kHz, audible);
               board streams 10x slower so the detections line up with what you hear.
--fake runs without a board: detections come from the bit-exact Python model of the FPGA
(identical output to the board), replayed in time with the audio. Good for rehearsing.
"""
import argparse, glob, json, os, queue, re, sys, threading, time, wave
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 256_000
CSV = re.compile(r"^\d+(,\d+){10}$")

NAMES = {  # class code -> display name (anything missing shows the code)
    "CNESER": "Serotine", "MINSCH": "Schreibers' bent-winged bat", "MYOCAP": "Long-fingered bat",
    "NYCNOC": "Common noctule", "PIPKUH/PIPNAT": "Kuhl's / Nathusius' pipistrelle",
    "PIPKUH": "Kuhl's pipistrelle", "PIPNAT": "Nathusius' pipistrelle",
    "PIPPIP": "Common pipistrelle", "RHIBLA": "Blasius's horseshoe bat",
    "RHIEUR": "Mediterranean horseshoe bat", "RHIFER": "Greater horseshoe bat",
    "RHIHIP": "Lesser horseshoe bat", "RHIMEH": "Mehely's horseshoe bat",
    "OTHER": "Other bat", "NOISE": "Noise (no bat)",
}


# ---------------------------------------------------------------- media pre-rendering
def read_wav(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, f"{path}: need 16-bit"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            x = x[:: w.getnchannels()]
        return x.astype(np.float32), w.getframerate()


def write_wav48(path, y):
    y = y / (np.max(np.abs(y)) or 1.0) * 0.7                  # playback level only
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def pitch_down(x, div=10, fs_out=48000):
    """Frequency division: keep the timing, divide every call's frequency by `div`.
    Tracks the strongest 15-125 kHz component every 0.25 ms and re-synthesises it as a tone at
    f/div (1.5-12.5 kHz for div=10), loudness from its level above the clip's noise floor."""
    hop, n = 64, 256
    m = (len(x) - n) // hop
    if m <= 0:
        return np.zeros(int(len(x) / FS * fs_out))
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1))[:, 15:126]
    k = S.argmax(axis=1)
    pk = S[np.arange(m), k]
    a, b, c = S[np.arange(m), np.maximum(k - 1, 0)], pk, S[np.arange(m), np.minimum(k + 1, S.shape[1] - 1)]
    off = np.clip(0.5 * (a - c) / np.where(a - 2 * b + c == 0, -1e-9, a - 2 * b + c), -0.5, 0.5)
    f = (15 + k + off) * 1000.0 / div                          # Hz after division
    db = 20 * np.log10(pk + 1e-9)
    amp = np.clip((db - np.median(db) - 8) / 25, 0, 1) ** 1.5  # gate the noise floor
    t_frames = (hop * np.arange(m) + n / 2) / FS
    t_out = np.arange(int(len(x) / FS * fs_out)) / fs_out
    fo = np.interp(t_out, t_frames, f)
    ao = np.interp(t_out, t_frames, amp)
    return ao * np.sin(2 * np.pi * np.cumsum(fo) / fs_out)


def render(clip, cache):
    """spectrogram PNG + audible renders, cached by file mtime."""
    base = os.path.splitext(os.path.basename(clip))[0]
    out = {k: os.path.join(cache, f"{base}{k}") for k in ("_spec.png", "_real.wav", "_pitch.wav", "_slow.wav")}
    x, sr = read_wav(clip)
    if sr != FS:
        print(f"WARNING {clip}: {sr} Hz, expected {FS}")
    info = dict(name=os.path.basename(clip), base=base, dur=len(x) / FS, frames=len(x) // 256)
    if all(os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(clip) for p in out.values()):
        return info
    from scipy.signal import resample_poly
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(x) // 256                                         # FPGA frames: 256 samples, 1 kHz bins
    fr = x[: n * 256].reshape(n, 256) * np.hanning(256)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1))[:, :128].T + 1e-3)
    S = S - np.median(S, axis=1, keepdims=True)              # per-frequency noise floor -> 0 dB
    if S.shape[1] > 4000:                                    # browsers: keep the image <= 4000 px
        k = int(np.ceil(S.shape[1] / 4000))
        S = np.pad(S, ((0, 0), (0, (-S.shape[1]) % k)), constant_values=S.min())
        S = S.reshape(S.shape[0], -1, k).max(axis=2)          # max-pool keeps short calls visible
    hi = max(np.percentile(S, 99.7), 6.0)
    plt.imsave(out["_spec.png"], np.clip(S / hi, 0, 1)[::-1] ** 0.8, cmap="magma")
    write_wav48(out["_real.wav"], resample_poly(x, 3, 16))    # 256k -> 48k: ultrasound removed
    write_wav48(out["_slow.wav"], resample_poly(x, 15, 8))    # as if 25.6 kHz -> 48k: x10 slower
    write_wav48(out["_pitch.wav"], pitch_down(x))            # same timing, frequencies / 10
    return info


def load_truth(clips_dir, info):
    """<clip>_truth.csv next to the clip (demo_truth.csv for audio.wav); single-species clips:
    species from the file name."""
    tf = os.path.join(clips_dir, "demo_truth.csv" if info["name"] == "audio.wav" else info["base"] + "_truth.csv")
    if os.path.exists(tf):
        rows = open(tf).read().splitlines()[1:]
        return [dict(s=int(r.split(",")[0]), e=int(r.split(",")[1]), species=r.split(",")[2].replace("_", " "),
                     cls=r.split(",")[3]) for r in rows if r.count(",") >= 3]
    m = re.match(r"([A-Z][a-z]+_[a-z]+)_CSE-([A-Z]+)-", info["name"])
    if m:
        return [dict(s=0, e=info["frames"], species=m.group(1).replace("_", " "), cls=m.group(2))]
    return []


def load_classes(path):
    cl = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            cl[int(p[0])] = dict(code=p[1], name=NAMES.get(p[1], p[1]), risk=p[2] if len(p) > 2 else "other_bat")
    return cl


# ---------------------------------------------------------------- board link
class Board:
    def __init__(self, port, board_dir, launch, broadcast):
        import serial
        self.s = serial.Serial(port, 115200, timeout=0.2)
        self.board_dir, self.broadcast = board_dir, broadcast
        self.ready = False
        self.buf = ""
        self.lock = threading.Lock()
        threading.Thread(target=self.reader, daemon=True).start()
        if launch:
            threading.Thread(target=self.launch, daemon=True).start()

    def send(self, text):
        with self.lock:
            self.s.write(text.encode())

    def launch(self):
        self.status("connecting to board...")
        self.send("\x03\n")                                   # stop anything running
        time.sleep(1.0)
        if "login:" in self.buf[-200:]:
            self.status("logging in..."); self.send("xilinx\n"); time.sleep(1.0); self.send("xilinx\n"); time.sleep(3)
        if "# " not in self.buf[-50:]:                        # not root yet
            self.send("sudo -i\n"); time.sleep(1.5)
            if "password" in self.buf[-200:].lower():
                self.send("xilinx\n"); time.sleep(2)
        self.status("loading FPGA overlay (10-30 s)...")
        self.send(f"cd {self.board_dir} && python3 run_bat.py --serve\n")

    def status(self, msg):
        print("[board]", msg)
        self.broadcast(dict(type="status", msg=msg, ready=self.ready))

    def reader(self):
        line = ""
        while True:
            try:
                data = self.s.read(256).decode(errors="ignore")
            except Exception as e:
                self.status(f"serial error: {e}"); time.sleep(1); continue
            if not data:
                continue
            self.buf = (self.buf + data)[-4000:]
            for ch in data:
                if ch in "\r\n":
                    if line.strip():
                        self.handle(line.strip())
                    line = ""
                else:
                    line += ch

    def handle(self, line):
        if CSV.match(line):
            v = list(map(int, line.split(",")))
            self.broadcast(dict(type="call", frame=v[0], cls=v[1], conf=v[2], dur=v[3], f_start=v[4],
                                f_end=v[5], f_peak=v[6], f_min=v[7], f_max=v[8], bw=v[9], sweep=v[10]))
        elif line == "READY":
            self.ready = True; self.status("FPGA ready")
        elif line.startswith("START "):
            p = line.split(); self.broadcast(dict(type="start", clip=p[1], speed=float(p[2])))
        elif line.startswith("END "):
            self.broadcast(dict(type="end", calls=int(line.split()[1])))
        elif line.startswith("ERROR") or "Traceback" in line or "Error" in line:
            self.status(line)
        else:
            print("[board]", line)

    def play(self, clip, speed):
        self.send(f"PLAY {clip} {speed:g}\n")

    def stop(self):
        self.send("STOP\n")


class FakeBoard:
    """No hardware: computes what the FPGA would output with the bit-exact Python model
    (fixed-point FFT + detector + tree.hex walk) and replays it in time with the audio."""
    def __init__(self, clips_dir, broadcast):
        self.broadcast, self.ready, self.clips_dir = broadcast, True, clips_dir
        self.cache = {}

    def events(self, clip):
        if clip not in self.cache:
            sys.path.insert(0, os.path.join(HERE, ".."))
            from bat_model import frame_stats, detect
            from fft_model import fft_mags_fixed
            tab = [int(l, 16) for l in open(os.path.join(self.clips_dir, "tree.hex")) if l.strip()]
            def walk(f):                                       # same as tree_walker.v
                n = 0
                while not tab[n] >> 31:
                    w = tab[n]
                    n = (w & 0x1FF) if f[(w >> 28) & 7] <= ((w >> 20) & 0xFF) else ((w >> 9) & 0x1FF)
                return (tab[n] >> 8) & 0xF, tab[n] & 0xFF
            x, _ = read_wav(os.path.join(self.clips_dir, clip))
            evs = []
            for st, en, f in detect(*frame_stats(fft_mags_fixed(x.astype(np.int16)))):
                c, conf = walk(f)
                evs.append([en, c, conf] + list(f))
            self.cache[clip] = evs
        return self.cache[clip]

    def stop(self):
        self.run_id = getattr(self, "run_id", 0) + 1

    def play(self, clip, speed):
        self.stop()
        me = self.run_id
        def run():
            evs = self.events(clip)
            self.broadcast(dict(type="start", clip=clip, speed=speed))
            t0 = time.time()
            for v in evs:
                time.sleep(max(0, v[0] / 1000 * speed - (time.time() - t0)))
                if self.run_id != me:
                    return
                self.broadcast(dict(type="call", frame=v[0], cls=v[1], conf=v[2], dur=v[3], f_start=v[4],
                                    f_end=v[5], f_peak=v[6], f_min=v[7], f_max=v[8], bw=v[9], sweep=v[10]))
            self.broadcast(dict(type="end", calls=len(evs)))
        threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- web server
def main():
    try:                                    # Windows console: never crash on odd characters
        sys.stdout.reconfigure(errors="replace"); sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port of the board (Mac /dev/tty.usbserial-XXXX1, Windows COM5)")
    ap.add_argument("--clips", default=os.path.join(HERE, "..", "sdcard", "batfpga_eu"),
                    help="local copy of the board folder (wavs, classes.txt, demo_truth.csv)")
    ap.add_argument("--board-dir", default="/home/xilinx/batfpga_eu", help="folder on the board")
    ap.add_argument("--no-launch", action="store_true", help="run_bat.py --serve already running on the board")
    ap.add_argument("--fake", action="store_true", help="no board: replay expected events (testing)")
    ap.add_argument("--http", type=int, default=8000)
    a = ap.parse_args()
    if not a.fake and not a.port:
        sys.exit("give --port (or --fake to try the UI without a board)")

    cache = os.path.join(HERE, "media"); os.makedirs(cache, exist_ok=True)
    clips = []
    for p in sorted(glob.glob(os.path.join(a.clips, "*.wav"))):
        info = render(p, cache)
        info["truth"] = load_truth(a.clips, info)
        clips.append(info)
        print(f"clip {info['name']}: {info['dur']:.2f} s")
    clips.sort(key=lambda c: (c["name"] != "demo_mix.wav", c["name"] != "audio.wav", c["name"]))
    classes = load_classes(os.path.join(a.clips, "classes.txt"))

    clients, clients_lock = [], threading.Lock()
    def broadcast(msg):
        with clients_lock:
            for q in clients:
                q.put(msg)

    board = FakeBoard(a.clips, broadcast) if a.fake else Board(a.port, a.board_dir, not a.no_launch, broadcast)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_bytes(self, data, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send_bytes(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            if path == "/api/state":
                st = dict(ready=board.ready, fake=a.fake, clips=clips, classes=classes)
                return self.send_bytes(json.dumps(st).encode(), "application/json")
            if path.startswith("/media/"):
                f = os.path.join(cache, os.path.basename(path))
                if os.path.exists(f):
                    return self.send_bytes(open(f, "rb").read(), "image/png" if f.endswith(".png") else "audio/wav")
            if path == "/api/events":
                q = queue.Queue()
                with clients_lock:
                    clients.append(q)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(f"data: {json.dumps(dict(type='status', msg='connected', ready=board.ready))}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=10)
                            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with clients_lock:
                        clients.remove(q)
                return
            self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path == "/api/stop":
                board.stop()
                return self.send_bytes(b"{}", "application/json")
            if urlparse(self.path).path == "/api/play":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                board.play(os.path.basename(body["clip"]), float(body.get("speed", 1)))
                return self.send_bytes(b"{}", "application/json")
            self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", a.http), H)
    print(f"\n  demo UI: http://localhost:{a.http}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
