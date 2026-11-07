# batfpga: low-power UK bat classifier (proof of concept)

The trained model, the RTL and the test vectors in this folder all come from **synthetic** bat calls (`synth_bats.py`). Swap in real ChiroVox recordings when they arrive (see the section on real data below).

## Pipeline: what runs where

```
int16 audio @256 kHz
  └─ 256-pt frames (1 ms), Hann window, FFT, mag=|re|+|im|      ← rtl/fft256.v (model: fft_model.py)
       └─ peak bin + peak magnitude over 15–125 kHz per frame    ← rtl/bat_features.v
            └─ call detector (adaptive noise floor)               ← rtl/bat_features.v
                 └─ 8 integer features per call                   ← rtl/bat_features.v
                      └─ decision tree → class + confidence       ← rtl/bat_tree.v (generated)
```
Features: duration, start/end/peak/min/max frequency (kHz), bandwidth, sweep.

`bat_model.py` is the integer reference model. Everything after the FFT has to match the RTL exactly.

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn matplotlib soundfile
.venv/bin/python train.py   # synth data → features → tree vs RF/boosting → tree.pkl, confusion_matrix.png
make golden                 # ROMs + rtl/bat_tree.v + golden vectors (export_fpga.py, gen_fft_rom.py)
make sim                    # post-FFT RTL vs Python on golden FFT mags          → PASS (bit-exact)
make sim_top                # audio → fft256 → bat_features vs fixed-point model → PASS (bit-exact)
make lint
```
`make sim` takes FFT magnitudes from Python, so it checks only the post-FFT logic. `make sim_top` runs the whole chain from int16 audio.

## FFT core (rtl/fft256.v)
This is an iterative radix-2 FFT built for low power. It has one butterfly unit (4 multipliers) and one 256×44-bit dual-port RAM. Each frame takes 256 load cycles, 3072 butterfly cycles and 128 output cycles, about 3456 cycles in total. Real time therefore needs only about **3.5 MHz**: at 100 MHz the core runs about 29× faster than real time and sits idle the rest of the time.
- `fft_model.py` is a bit-exact integer model of the core: 4 guard bits, Q16 twiddles, a rounded /2 at every stage. It stays within ±2 LSB of the float `bat_model.fft_mags()`.
- On the synthetic golden stream, events from the RTL FFT match the float-FFT reference 46/46, with every feature identical.
- `in_ready` goes low while a frame is computing. That is fine for audio fed from BRAM/DDR, but a live 256 kHz ADC needs a FIFO of at least 256 samples in front.

## Results on synthetic data (held-out recordings, never seen in training)
| model | species acc | risk-group acc |
|---|---|---|
| decision tree d8 (FPGA, 56 leaves) | 0.936 | 0.946 |
| random forest 100 | 0.951 | 0.956 |
| gradient boosting | 0.941 | 0.949 |

The single small tree comes within about 1.5 points of the big ensembles. The main confusion is between Daubenton's and whiskered bat, which is the same Myotis problem real tools have. On those calls the confidence output drops (about 140–210 out of 255), which gives the "flag for expert review" behaviour.

## Next steps
1. **Electronics:** FFT done (`rtl/fft256.v`). Next: PYNQ-Z2 bring-up, with audio in BRAM/DDR → `bat_top` → UART.
2. **Real data:** write a loader that returns `dict(cls, rec_id, x=int16 @256 kHz, intervals=[])` per ChiroVox recording. Resample to 256 kHz after correcting time-expanded files. In `bat_model.extract`, label events by the recording's species, not by `intervals`. Then rerun `train.py` and `export_fpga.py`.
3. **Biology:** `confusion_matrix.csv` holds the sensor error model for the outbreak simulation.
4. **Tune:** your bat teammate should check the species parameters in `synth_bats.py`. The detector thresholds are at the top of `bat_model.py`.

Class ids and risk groups are listed in `golden/classes.txt` (0 = Daubenton's / EBLV-2, 1 = serotine / EBLV-1).
