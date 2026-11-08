# Live demo: laptop plays the bat audio, the FPGA classifies it

The laptop plays a recording out loud. At the same moment the board streams **the same recording from its SD card** through the FPGA. Each call the FPGA classifies appears on the spectrogram and in the live feed, coloured by disease-risk group. No Ethernet is needed: the laptop and board talk over the micro-USB serial link.

## One-time setup
1. Run `make sdcard SD=batfpga_eu` on the Mac. Copy `sdcard/batfpga_eu/` to the SD card, then on the board copy it to `/home/xilinx/batfpga_eu`: run `sudo -i`, mount the card, then `cp -r /mnt/sd/batfpga_eu /home/xilinx/`. The folder needs the **new** `run_bat.py`, which has `--serve`.
2. On the laptop:
   ```bash
   cd batfpga
   .venv/bin/pip install pyserial numpy scipy matplotlib     # or: pip install ...
   ```
   The laptop also needs a local copy of the same folder, which is already in `sdcard/batfpga_eu/` on the Mac.

## Running it
```bash
cd batfpga
.venv/bin/python demo/demo_ui.py --fake                              # no board (rehearsal)
.venv/bin/python demo/demo_ui.py --port /dev/tty.usbserial-XXXX1     # Mac + board
.venv\Scripts\python demo\demo_ui.py --port COM5                     # Windows + board
```
Open **http://localhost:8000**. With the board, close PuTTY/screen first and wait for the green **FPGA ready** badge (10–30 s while the overlay loads). In `--fake` mode the detections come from the bit-exact Python model of the FPGA, so they're identical to the board's output.

**Controls:** **Play/Stop** (Space) · listening mode **As recorded / Pitched down / Slowed down** (keys 1/2/3) · **FPGA classifier** switch (C). The mode and the switch apply to the next Play.

## Suggested flow
1. **Hidden frequencies (classifier off).** Select **Demo mix (15 s)**.
   - Play in **As recorded**: "That's a real recording of 12 European bat species. You heard almost nothing." The spectrogram fills in as it plays: "Everything above the dashed line is above human hearing."
   - Play in **Pitched down**: every frequency is divided by 10 with the timing kept, so now you hear the calls, in step with the picture. "Each species has its own call shape: horseshoe bats hold a constant tone at 80–110 kHz, pipistrelles sweep down to about 45 kHz."
2. **Why it matters:** switch tabs to the disease and simulation dashboard.
3. **Back to the demo: switch the FPGA classifier on** (C), then Play in **Pitched down**. "The FPGA is now streaming the same recording from its SD card in real time, and every box is a call it has classified." Red boxes and the banner mark SARS-related-coronavirus reservoir species, orange marks Lloviu virus, and purple marks lyssavirus. The ✓/✗ counter compares each answer with the species label of the held-out recording.
4. **How it works / low power:** FFT → detector → 8 integer features → depth-8 decision tree, all in FPGA logic. See the key facts below.

## Key facts (only quote measured numbers)
- The classifier logic is estimated at about **11 mW** (Vivado vectorless estimate, `overlay/reports/power_hier.txt`). The ARM processor that drives the demo is about 1.5 W and wouldn't be needed in a field sensor.
- It needs about **4,480 clock cycles per 1 ms of audio**, so real time takes only about 4.5 MHz. Quote the cycles/frame line `run_bat.py` prints on the board.
- The board output is **bit-exact** with the Python model (self-test PASS on real hardware).
- On the held-out demo mix it labels **54/74 calls** correctly (real data).

## How a field sensor would listen continuously
The FPGA pipeline already processes an endless stream: 1 ms frames, with an adaptive noise floor in the detector. It has no notion of a "file". In the field:
- An ultrasonic MEMS microphone and ADC at 256 kHz feed a small FIFO, replacing the DMA from the SD card.
- The FPGA logic runs continuously at a few MHz. No ARM, no Linux, no storage of raw audio.
- Only detections leave the device, as a few bytes per call, sent over LoRa or similar to raise an alert when a reservoir species is present. Counting calls per species over minutes turns per-call labels into a robust presence signal.

Recording-level voting smooths out single-call mistakes. A NOISE class (in progress) rejects insects, rain and wind.
