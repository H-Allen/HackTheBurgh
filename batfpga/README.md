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
make sim_tree               # programmable tree (tree.hex) == generated bat_tree.v, 200k vectors
make sim_axi                # PYNQ wrapper, same register sequence as board/run_bat.py
make test                   # all four
make lint
```
`make sim` takes FFT magnitudes from Python, so it checks only the post-FFT logic. `make sim_top` runs the whole chain from int16 audio.

## FFT core (rtl/fft256.v)
This is an iterative radix-2 FFT built for low power. It has one butterfly unit (4 multipliers) and one 256×44-bit dual-port RAM. Each frame takes 256 load cycles, 3072 butterfly cycles and 128 output cycles, about 3456 cycles in total. Real time therefore needs only about **3.5 MHz**: at 100 MHz the core runs about 29× faster than real time and sits idle the rest of the time.
- `fft_model.py` is a bit-exact integer model of the core: 4 guard bits, Q16 twiddles, a rounded /2 at every stage. It stays within ±2 LSB of the float `bat_model.fft_mags()`.
- On the synthetic golden stream, events from the RTL FFT match the float-FFT reference 46/46, with every feature identical.
- `in_ready` goes low while a frame is computing. That is fine for audio fed from BRAM/DDR, but a live 256 kHz ADC needs a FIFO of at least 256 samples in front.

## PYNQ-Z2 board
The decision tree is **programmable**: `tree_walker.v` walks a node table (`golden/tree.hex`), and `run_bat.py` loads that table at start-up. After retraining you only copy a new `tree.hex`. The bitstream does not need rebuilding.

**1. Build the bitstream (lab machine with Vivado, once):**
```bash
git clone … && cd batfpga
vivado -mode batch -source vivado/build.tcl     # → overlay/bat.bit, overlay/bat.hwh, overlay/reports/
```
This builds the PS7, an AXI DMA (MM2S) and `bat_axi.v` at FCLK0 = 50 MHz (real time needs about 3.5 MHz). Commit or copy the `overlay/` folder. The script has not been run in Vivado yet, so expect to fix small version-specific Tcl issues on the first run.

**2. Prepare the SD card (any laptop):** run `make sdcard`, then copy `sdcard/batfpga/` onto the SD card's FAT boot partition, or onto a USB stick plugged into the board. The folder holds `bat.bit`, `bat.hwh`, `run_bat.py`, `tree.hex`, `classes.txt`, demo audio and the expected events.

**3. Run it** over the micro-USB serial console (115200 baud; `screen` on Mac, PuTTY on Windows; login `xilinx`/`xilinx`):
```bash
sudo -i
cd /boot/batfpga          # or wherever the FAT partition / USB stick is mounted (lsblk)
python3 run_bat.py audio.hex --expect expected_events_fixedfft.txt    # self-test → PASS
python3 run_bat.py audio.wav --realtime                                # demo
```
- **stdout** carries one CSV line per call (the teammate interface).
- **stderr** carries species names, FPGA cycles per frame, the "×-faster-than-real-time" figure and wall-clock time.
- On the laptop, `board/serial_reader.py PORT calls.csv` keeps only the CSV lines, for the dashboard.

Register map: see the header of `rtl/bat_axi.v`.

## Results on synthetic data (held-out recordings, never seen in training)
| model | species acc | risk-group acc |
|---|---|---|
| decision tree d8 (FPGA, 56 leaves) | 0.936 | 0.946 |
| random forest 100 | 0.951 | 0.956 |
| gradient boosting | 0.941 | 0.949 |

The single small tree comes within about 1.5 points of the big ensembles. The main confusion is between Daubenton's and whiskered bat, which is the same Myotis problem real tools have. On those calls the confidence output drops (about 140–210 out of 255), which gives the "flag for expert review" behaviour.

## Next steps
1. **Electronics:** run `vivado/build.tcl` on a lab machine, then do the board self-test (see the PYNQ-Z2 section above).
2. **Real data:** write a loader that returns `dict(cls, rec_id, x=int16 @256 kHz, intervals=[])` per ChiroVox recording. Resample to 256 kHz after correcting time-expanded files. In `bat_model.extract`, label events by the recording's species, not by `intervals`. Then rerun `train.py` and `export_fpga.py`.
3. **Biology:** `confusion_matrix.csv` holds the sensor error model for the outbreak simulation.
4. **Tune:** your bat teammate should check the species parameters in `synth_bats.py`. The detector thresholds are at the top of `bat_model.py`.

Class ids and risk groups are listed in `golden/classes.txt` (0 = Daubenton's / EBLV-2, 1 = serotine / EBLV-1).
