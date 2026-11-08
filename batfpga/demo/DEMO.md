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
1. Power the board and wait about a minute for it to boot. Close `screen`/PuTTY, because only one program can use the serial port.
2. Start the demo:
   ```bash
   .venv/bin/python demo/demo_ui.py --port /dev/tty.usbserial-XXXX1     # Windows: --port COM5
   ```
   It logs in (xilinx/xilinx) if needed, becomes root, and starts `run_bat.py --serve` on the board.
3. Open **http://localhost:8000**. Wait until the top-right badge says **FPGA ready** (10–30 s while the overlay loads).
4. Pick a recording. **"All species (demo mix)"** is the best one to start with.

To try the UI with no board connected, run `demo_ui.py --fake`. The detections are then replayed from the Python model.

## Suggested script
1. **▶ Real time.** "This is a real recording of horseshoe bats, at real speed. You can't hear anything, because the calls are ultrasonic, 20–110 kHz. But the FPGA can: each box is a call it has just detected and classified, in real time."
2. **▶ 10× slowed.** "Now slowed down ten times, which is the time-expansion trick bat detectors use. Now you can hear the chirps, and each one lights up as the FPGA classifies it." Red boxes and the alert banner mark species that are SARS-related-coronavirus reservoirs.
3. Point at the tick/cross counter: the FPGA's answers compared with the species label of each held-out recording.

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
