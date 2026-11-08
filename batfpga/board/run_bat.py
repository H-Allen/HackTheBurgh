#!/usr/bin/env python3
"""
run_bat.py - run the bat classifier overlay on the PYNQ-Z2 and print one CSV line per call.

On the board (serial console):
    sudo -i                         # PYNQ's Python + pynq package live in root's environment
    cd /path/to/batfpga_sd          # folder with bat.bit, bat.hwh, tree.hex, audio, this script
    python3 run_bat.py audio.wav                      # as fast as possible
    python3 run_bat.py audio.wav --realtime           # paced to real time (live-looking demo)
    python3 run_bat.py audio.hex --expect expected_events_fixedfft.txt   # self-test
    python3 run_bat.py audio.wav --truth demo_truth.csv --realtime       # demo with ground truth
    python3 run_bat.py --serve      # demo UI mode: load overlay once, then take commands on stdin:
        PLAY <file.wav> [speed]     stream file paced at real time x speed (10 = 10x slowed down)
        STOP                        abort the current PLAY
        QUIT
      prints READY, START <file> <speed> <samples>, CSV call lines, STATS <samples> <cycles> <fft_busy>
      <fclk_MHz> (hardware counters), END <calls>

stdout: frame_idx,class_id,confidence,dur,f_start,f_end,f_peak,f_min,f_max,bw,sweep
stderr: species names, stats, PASS/FAIL (so stdout stays clean CSV for the dashboard).
Audio: mono int16 at 256 kHz (.wav, .npy, or .hex with one 16-bit hex word per line).
"""
import argparse, os, sys, time, wave
import numpy as np

CTRL, STATUS, EV_FRAME, EV_CLASS, EV_LO, EV_HI, EV_POP, ID = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C
TREE_ADDR, TREE_DATA, CYCLES, SAMPLES, BUSY = 0x20, 0x24, 0x28, 0x2C, 0x30
FS = 256000
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_audio(path):
    if path.endswith(".wav"):
        with wave.open(path) as w:
            if w.getsampwidth() != 2:
                sys.exit("wav must be 16-bit")
            if w.getframerate() != FS:
                log(f"WARNING: wav is {w.getframerate()} Hz, pipeline expects {FS} Hz (resample offline)")
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            if w.getnchannels() > 1:
                x = x[:: w.getnchannels()]
        return x.astype(np.int16)
    if path.endswith(".npy"):
        return np.load(path).astype(np.int16)
    with open(path) as f:
        return np.array([int(l, 16) for l in f if l.strip()], dtype=np.uint16).view(np.int16)


def load_classes(path):
    names = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 2:
                names[int(p[0])] = " ".join(p[1:])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--serve", action="store_true", help="demo UI mode (see top of file)")
    ap.add_argument("--bit", default=os.path.join(HERE, "bat.bit"))
    ap.add_argument("--tree", default=os.path.join(HERE, "tree.hex"))
    ap.add_argument("--classes", default=os.path.join(HERE, "classes.txt"))
    ap.add_argument("--expect", help="expected events file (frame cls conf 8 feats) -> PASS/FAIL")
    ap.add_argument("--realtime", action="store_true", help="pace input to real time")
    ap.add_argument("--chunk", type=float, default=0.1, help="seconds of audio per DMA transfer")
    ap.add_argument("--out", help="also write CSV to this file")
    ap.add_argument("--header", action="store_true", help="print CSV header line first")
    ap.add_argument("--quiet", action="store_true", help="no per-call species line on stderr")
    ap.add_argument("--truth", help="demo_truth.csv (start_frame,end_frame,species,class): "
                                    "show true species next to each FPGA answer + accuracy")
    a = ap.parse_args()

    from pynq import Overlay, allocate, Clocks

    if not a.serve and not a.audio:
        sys.exit("give an audio file, or --serve")
    tree = [int(l, 16) for l in open(a.tree) if l.strip()]
    names = load_classes(a.classes)
    truth = []
    if a.truth:
        for l in open(a.truth).read().splitlines()[1:]:
            p = l.split(",")
            if len(p) >= 4:
                truth.append((int(p[0]), int(p[1]), p[2], p[3]))
    def true_of(frame):
        for s0, s1, sp, cl in truth:
            if s0 <= frame - 1 < s1:              # frame = first quiet frame after the call
                return sp, cl
        return None

    t = time.time()
    ol = Overlay(a.bit)
    dma, bat = ol.axi_dma, ol.bat_axi_0
    fclk = Clocks.fclk0_mhz
    log(f"overlay loaded in {time.time() - t:.1f} s, FCLK0 {fclk:.1f} MHz; tree: {len(tree)} nodes")
    if bat.read(ID) != 0xBA7F0001:
        sys.exit(f"bad ID register {bat.read(ID):08x}: wrong bitstream?")

    bat.write(CTRL, 1)                                # hold core in reset
    bat.write(TREE_ADDR, 0)
    for w in tree:
        bat.write(TREE_DATA, w)

    def reset_core():                                 # frame index restarts at 0, FIFO empty
        bat.write(CTRL, 1 | 2)
        bat.write(CTRL, 0)

    out = open(a.out, "w") if a.out else None
    def emit(line):
        print(line, flush=True)
        if out:
            out.write(line + "\n"); out.flush()
    if a.header:
        emit("frame_idx,class_id,confidence,dur,f_start,f_end,f_peak,f_min,f_max,bw,sweep")

    events = []
    def drain(quiet):
        st = bat.read(STATUS)
        if st & (1 << 16):
            log("WARNING: event FIFO overflowed (events lost)")
        for _ in range(st & 0x7FF):
            frame = bat.read(EV_FRAME); c = bat.read(EV_CLASS)
            lo = bat.read(EV_LO); hi = bat.read(EV_HI)
            bat.write(EV_POP, 1)
            f = [(lo >> (8 * k)) & 0xFF for k in range(4)] + [(hi >> (8 * k)) & 0xFF for k in range(4)]
            ev = (frame, (c >> 8) & 0xF, c & 0xFF, *f)
            events.append(ev)
            emit(",".join(map(str, ev)))
            if not quiet:
                pred = names.get(ev[1], str(ev[1]))
                t = true_of(frame) if truth else None
                if t:
                    ok = pred.split()[0] == t[1]
                    log(f"  t={frame / 1000:8.3f}s  true {t[0]:28s} FPGA {pred.split()[0]:14s} conf {ev[2]:3d}  "
                        f"{'OK' if ok else 'x'}")
                else:
                    log(f"  t={frame / 1000:8.3f}s  {pred}  conf {ev[2]}")

    bufs = {}
    def stop_requested():
        """serve mode: a STOP line on stdin aborts the current stream."""
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line or line.strip().upper() == "STOP":
                return True
        return False

    def stream(x, speed, chunk_s, quiet, can_stop=False):
        """DMA x through the FPGA in chunks; speed=None: as fast as possible, else paced so the
        chunk is sent once its audio 'has happened' at real time x speed."""
        chunk = max(2, int(chunk_s * FS) & ~1)
        t0 = time.time(); hw_s = 0.0
        for i in range(0, len(x), chunk):
            seg = x[i:i + chunk]
            if len(seg) not in bufs:
                bufs[len(seg)] = allocate(shape=(len(seg),), dtype=np.int16)
            buf = bufs[len(seg)]
            buf[:] = seg
            buf.flush()
            if speed:
                lag = (i + len(seg)) / FS * speed - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            if can_stop and stop_requested():
                break
            ts = time.time()
            dma.sendchannel.transfer(buf)
            dma.sendchannel.wait()
            while not bat.read(STATUS) & (1 << 18):   # core idle: last frame processed
                pass
            hw_s += time.time() - ts
            drain(quiet)
        time.sleep(0.001); drain(quiet)
        return time.time() - t0, hw_s

    if a.serve:
        import glob
        cache = {}
        for f in sorted(glob.glob(os.path.join(HERE, "*.wav"))):     # preload: no stall on first PLAY
            try:
                xx = load_audio(f); cache[f] = xx[: len(xx) & ~1]
            except Exception as e:
                log(f"skip {f}: {e}")
        log(f"preloaded {len(cache)} clips")
        print("READY", flush=True)
        for line in sys.stdin:
            p = line.split()
            if not p:
                continue
            if p[0].upper() == "QUIT":
                break
            if p[0].upper() == "STOP":
                continue                              # nothing playing
            if p[0].upper() != "PLAY" or len(p) < 2:
                print(f"ERROR bad command: {line.strip()}", flush=True); continue
            path = p[1] if os.path.exists(p[1]) else os.path.join(HERE, p[1])
            speed = float(p[2]) if len(p) > 2 else 1.0
            try:
                if path not in cache:
                    xx = load_audio(path)
                    cache[path] = xx[: len(xx) & ~1]
            except Exception as e:
                print(f"ERROR {e}", flush=True); continue
            x = cache[path]
            reset_core(); events.clear()
            print(f"START {os.path.basename(path)} {speed:g} {len(x)}", flush=True)
            stream(x, speed, 0.02, quiet=True, can_stop=True)
            smp, cyc, busy = bat.read(SAMPLES), bat.read(CYCLES), bat.read(BUSY)
            print(f"STATS {smp} {cyc} {busy} {fclk:.2f}", flush=True)   # read from the FPGA's registers
            print(f"END {len(events)}", flush=True)
        return

    x = load_audio(a.audio)
    x = x[: len(x) & ~1]                              # DMA words carry 2 samples
    log(f"audio: {len(x)} samples = {len(x) / FS:.2f} s")
    reset_core()
    wall, hw_s = stream(x, 1.0 if a.realtime else None, a.chunk, a.quiet)

    smp, busy, cyc = bat.read(SAMPLES), bat.read(BUSY), bat.read(CYCLES)
    frames = smp // 256
    audio_s = smp / FS
    per_frame = (busy / frames + 256) if frames else 0     # compute + 256 load cycles
    log("-" * 60)
    log(f"{len(events)} calls in {audio_s:.2f} s of audio ({frames} frames)")
    log(f"FPGA work: {per_frame:.0f} cycles/frame -> real time needs {per_frame * 1000 / 1e6:.2f} MHz; "
        f"at {fclk:.0f} MHz = {fclk * 1e6 / (per_frame * 1000):.1f}x faster than real time")
    log(f"FFT busy {100 * busy / max(cyc, 1):.1f}% of cycles (rest idle waiting for samples)")
    log(f"wall clock incl. DMA + Python: {wall:.2f} s total, {hw_s:.2f} s in DMA+compute "
        f"({audio_s / max(hw_s, 1e-9):.1f}x real time)")

    if truth:
        hits = [(true_of(e[0]), names.get(e[1], str(e[1])).split()[0]) for e in events]
        hits = [(t, p) for t, p in hits if t]
        if hits:
            log(f"labels vs demo_truth: {sum(t[1] == p for t, p in hits)}/{len(hits)} calls correct "
                f"(species), {len({t[0] for t, p in hits})} species in the demo")

    if a.expect:
        exp = [tuple(int(v) for v in l.split()) for l in open(a.expect) if l.strip()]
        ok = exp == events
        log(f"self-test vs {a.expect}: {len(events)} got / {len(exp)} expected -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            for i, (e, g) in enumerate(zip(exp, events)):
                if e != g:
                    log(f"  first mismatch #{i}: exp {e}  got {g}"); break
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
