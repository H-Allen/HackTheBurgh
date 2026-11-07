# batfpga: low-power UK bat classifier (proof of concept)

The trained model, the RTL and the test vectors in this folder all come from **synthetic** bat calls (`synth_bats.py`). Swap in real ChiroVox recordings when they arrive (see the section on real data below).

## Pipeline: what runs where

```
int16 audio @256 kHz
  └─ 256-pt frames (1 ms), Hann window, FFT, mag=|re|+|im|      ← FFT core (ZipCPU / Xilinx IP)
       └─ peak bin + peak magnitude over 15–125 kHz per frame    ← rtl/bat_features.v
            └─ call detector (adaptive noise floor)               ← rtl/bat_features.v
                 └─ 8 integer features per call                   ← rtl/bat_features.v
                      └─ decision tree → class + confidence       ← rtl/bat_tree.v (generated)
```
Features: duration, start/end/peak/min/max frequency (kHz), bandwidth, sweep.

`bat_model.py` is the integer reference model. Everything after the FFT has to match the RTL exactly.

## Run it
```bash
pip install numpy scipy scikit-learn matplotlib
python train.py          # synth data → features → tree vs RF/boosting → tree.pkl, confusion_matrix.png
python export_fpga.py    # tree.pkl → rtl/bat_tree.v ; golden/bins.hex + expected_events.txt

verilator --cc --exe --build -O2 -Wno-fatal -Irtl --top-module bat_features \
    rtl/bat_features.v rtl/bat_tree.v sim/tb_bat.cpp -o tb_bat
./obj_dir/tb_bat golden/bins.hex golden/expected_events.txt     # expect: PASS
```
The testbench takes the FFT magnitudes from Python, so the post-FFT logic can be verified before the FFT core exists.

## Results on synthetic data (held-out recordings, never seen in training)
| model | species acc | risk-group acc |
|---|---|---|
| decision tree d8 (FPGA, 56 leaves) | 0.936 | 0.946 |
| random forest 100 | 0.951 | 0.956 |
| gradient boosting | 0.941 | 0.949 |

The single small tree comes within about 1.5 points of the big ensembles. The main confusion is between Daubenton's and whiskered bat, which is the same Myotis problem real tools have. On those calls the confidence output drops (about 140–210 out of 255), which gives the "flag for expert review" behaviour.

## Next steps
1. **Electronics:** wire up an FFT core in front of `bat_features`. Use a 256-point FFT with a Hann window and scale the output by 1/256 (>>8) to match `fft_mags()`. Then feed it audio from BRAM/DDR. Exact bit-for-bit match isn't needed at the FFT stage, because the tree thresholds tolerate small differences.
2. **Real data:** write a loader that returns `dict(cls, rec_id, x=int16 @256 kHz, intervals=[])` per ChiroVox recording. Resample to 256 kHz after correcting time-expanded files. In `bat_model.extract`, label events by the recording's species, not by `intervals`. Then rerun `train.py` and `export_fpga.py`.
3. **Biology:** `confusion_matrix.csv` holds the sensor error model for the outbreak simulation.
4. **Tune:** your bat teammate should check the species parameters in `synth_bats.py`. The detector thresholds are at the top of `bat_model.py`.

Class ids and risk groups are listed in `golden/classes.txt` (0 = Daubenton's / EBLV-2, 1 = serotine / EBLV-1).
