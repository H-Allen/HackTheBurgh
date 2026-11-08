"""
make_demo.py - build demo / golden audio from HELD-OUT real recordings (manifest split == test).

  python make_demo.py ChirosetEurope_manifest_v1.csv --audio-dir audio --features features.csv
  -> golden/demo/<Genus_species>_<chiroset_id>.wav   one 256 kHz int16 clip per species (--seconds)
     golden/demo_golden.wav                          short concatenation (--golden-seconds per
                                                     species) used by export_fpga.py --audio for
                                                     the board self-test vectors
     golden/demo_truth.csv                           ground truth for demo_golden.wav:
                                                     start_frame,end_frame,species,class
     golden/demo/demo_mix.wav (+ demo_mix_truth.csv) live-demo mix: --mix-species, one natural
                                                     pass (--mix-per s) each, --mix-gap s silence
Preprocessing is identical to chiroset_features.py (sr_eff = sampling_rate x time expansion,
resample to 256 kHz, float -> int16, no normalisation), so the detector sees what training saw.
Only clips where bat_model's detector finds >= --min-calls calls are used.
"""
import argparse, os, pickle, wave
from fractions import Fraction
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from bat_model import fft_mags, frame_stats, detect

FS = 256_000


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes(x.astype("<i2").tobytes())


def load(path, sr_eff_manifest, seconds):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    te = sr_eff_manifest / sr if sr else 1.0
    sr_eff = int(round(sr * max(te, 1.0)))
    x = x[: int(seconds * sr)]
    fr = Fraction(FS, sr_eff).limit_denominator(1000)
    y = resample_poly(x, fr.numerator, fr.denominator) if sr_eff != FS else x
    return np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)


SHORT = {"Cnephaeus serotinus": "CNESER", "Rhinolophus ferrumequinum": "RHIFER",
         "Rhinolophus hipposideros": "RHIHIP", "Rhinolophus euryale": "RHIEUR",
         "Rhinolophus mehelyi": "RHIMEH", "Rhinolophus blasii": "RHIBLA",
         "Miniopterus schreibersii": "MINSCH"}


def class_of(latin, classes):
    """Latin name -> class name in tree.pkl (same codes as train_from_features.py; merged
    classes like 'PIPKUH/PIPNAT' match either code; anything else -> OTHER)."""
    g, s = (latin.split() + [""])[:2]
    c = SHORT.get(latin, (g[:3] + s[:3]).upper())
    for k in classes:
        if c in k.split("/"):
            return k
    return "OTHER" if "OTHER" in classes else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--features", default="features.csv", help="species list = species in here")
    ap.add_argument("--seconds", type=float, default=5.0, help="length of each demo clip")
    ap.add_argument("--golden-seconds", type=float, default=0.75, help="per species in demo_golden.wav")
    ap.add_argument("--min-calls", type=int, default=3)
    ap.add_argument("--min-sr", type=int, default=192_000)
    ap.add_argument("--out", default="golden/demo")
    ap.add_argument("--mix-species", default="Pipistrellus pipistrellus,Nyctalus noctula,Cnephaeus serotinus,"
                    "Miniopterus schreibersii,Rhinolophus ferrumequinum,Rhinolophus euryale",
                    help="comma-separated latin names, in play order (default covers every risk group)")
    ap.add_argument("--mix-per", type=float, default=2.5, help="seconds per species (one recording)")
    ap.add_argument("--mix-gap", type=float, default=1.2, help="seconds of silence between species")
    a = ap.parse_args()

    species = sorted(pd.read_csv(a.features, usecols=["species"]).species.dropna().unique())
    m = pd.read_csv(a.manifest, low_memory=False)
    m["sr_eff"] = m.sampling_rate * pd.to_numeric(m.time_expansion_factor, errors="coerce").fillna(1.0)
    m = m[(m.split == "test") & m.latin.isin(species) & (m.sr_eff >= a.min_sr)
          & m.voc_type.astype(str).str.lower().str.contains("echo|nan")]
    index = {}
    for root, _, files in os.walk(a.audio_dir):
        for fn in files:
            index.setdefault(fn.lower(), os.path.join(root, fn))

    classes = pickle.load(open("tree.pkl", "rb"))["classes"]
    os.makedirs(a.out, exist_ok=True)
    golden, truth, pos = [], [], 0
    for sp in species:
        rows = m[m.latin == sp].sample(frac=1, random_state=0)
        done = False
        for r in rows.itertuples():
            path = index.get(os.path.basename(str(r.file_name)).lower()) or \
                   index.get(os.path.basename(str(r.file_path)).lower())
            if not path:
                continue
            try:
                x = load(path, r.sr_eff, a.seconds)
            except Exception as e:
                print(f"  skip {path}: {e}"); continue
            n = len(detect(*frame_stats(fft_mags(x))))
            if n < a.min_calls:
                continue
            name = f"{sp.replace(' ', '_')}_{r.chiroset_id}.wav"
            write_wav(os.path.join(a.out, name), x)
            g = x[: int(a.golden_seconds * FS)]
            seg = np.concatenate([g, np.zeros(FS // 20, np.int16)])           # 50 ms gap
            golden.append(seg)
            truth.append((pos // 256, (pos + len(seg)) // 256, sp.replace(" ", "_"), class_of(sp, classes)))
            pos += len(seg)
            print(f"{sp:28s} {name}  {len(x) / FS:.1f} s, {n} calls (held-out test split)")
            done = True
            break
        if not done:
            print(f"{sp:28s} -- no held-out clip with >= {a.min_calls} calls found")
    if golden:
        g = np.concatenate(golden)
        write_wav("golden/demo_golden.wav", g)
        with open("golden/demo_truth.csv", "w") as f:
            f.write("start_frame,end_frame,species,class\n")
            f.writelines(f"{a_},{b_},{sp_},{c_}\n" for a_, b_, sp_, c_ in truth)
        print(f"golden/demo_golden.wav: {len(g) / FS:.1f} s from {len(golden)} species")

    # ---- live-demo mix: one natural pass per species (--mix-per s, densest window of a single
    # held-out recording, chosen by call activity only), --mix-gap s of silence in between ----
    want = [w.strip() for w in a.mix_species.split(",") if w.strip()]
    fade = np.minimum(1, np.minimum(np.arange(int(a.mix_per * FS)), np.arange(int(a.mix_per * FS))[::-1]) / (0.02 * FS))
    mix, mtruth, pos = [], [], 0
    gap = np.zeros(int(a.mix_gap * FS), np.int16)
    for sp in want:
        best = None
        for r in m[m.latin == sp].sample(frac=1, random_state=1).itertuples():
            path = index.get(os.path.basename(str(r.file_name)).lower()) or \
                   index.get(os.path.basename(str(r.file_path)).lower())
            if not path:
                continue
            try:
                x = load(path, r.sr_eff, 30.0)
            except Exception:
                continue
            if len(x) < a.mix_per * FS:
                continue
            ends = [e[1] for e in detect(*frame_stats(fft_mags(x)))]        # call end frames
            win = int(a.mix_per * 1000)
            nwin = len(x) // 256
            for st in range(0, max(1, nwin - win), 50):                       # densest window
                n = sum(st + 30 <= e < st + win - 30 for e in ends)
                if best is None or n > best[0]:
                    best = (n, x[st * 256: st * 256 + int(a.mix_per * FS)], r.chiroset_id)
            if best and best[0] >= 6:
                break
        if not best or best[0] < 3:
            # short recordings only: join several (with 0.2 s pauses) into one --mix-per slot
            parts, need, n, ids = [], int(a.mix_per * FS), 0, []
            for r in m[m.latin == sp].sample(frac=1, random_state=1).itertuples():
                path = index.get(os.path.basename(str(r.file_name)).lower()) or \
                       index.get(os.path.basename(str(r.file_path)).lower())
                if not path or need <= 0:
                    continue
                try:
                    x = load(path, r.sr_eff, 30.0)
                except Exception:
                    continue
                k = len(detect(*frame_stats(fft_mags(x))))
                if k == 0:
                    continue
                seg = np.concatenate([x[:need], np.zeros(FS // 5, np.int16)])
                parts.append(seg); need -= len(seg); n += k; ids.append(r.chiroset_id)
            if n < 3:
                print(f"mix: {sp} -- no held-out recordings with enough calls"); continue
            best = (n, np.concatenate(parts)[: int(a.mix_per * FS)], f"{len(ids)} short recordings")
        clip = (best[1].astype(np.float32) * fade[: len(best[1])]).astype(np.int16)
        if mix:
            mix.append(gap); pos += len(gap)
        mix.append(clip)
        mtruth.append((pos // 256, (pos + len(clip)) // 256, sp.replace(" ", "_"), class_of(sp, classes)))
        pos += len(clip)
        print(f"mix: {sp:28s} {len(clip) / FS:.1f} s from {best[2]} ({best[0]} calls)")
    if mix:
        g = np.concatenate(mix + [np.zeros(FS // 2, np.int16)])
        write_wav(os.path.join(a.out, "demo_mix.wav"), g)
        with open(os.path.join(a.out, "demo_mix_truth.csv"), "w") as f:
            f.write("start_frame,end_frame,species,class\n")
            f.writelines(f"{a_},{b_},{sp_},{c_}\n" for a_, b_, sp_, c_ in mtruth)
        print(f"{a.out}/demo_mix.wav: {len(g) / FS:.1f} s, {len(mtruth)} species")


if __name__ == "__main__":
    main()
