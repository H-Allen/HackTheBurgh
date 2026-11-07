# Board guide: Vivado build → SD card → PYNQ-Z2 test

There are four stages. Stage 0 runs alongside Stage 1. Everything here uses the **synthetic** model. Real data later only replaces `tree.hex` and the audio files, so the bitstream doesn't change.

---

## Stage 0: check the board boots (Windows laptop, ~15 min, do this now)

**Set up the board.** Put the SD card in. Set the boot jumper (**SD / QSPI / JTAG**) to **SD** and the power jumper (**USB / REG**) to **USB**. Plug micro-USB into **PROG/UART** → laptop.

**Install PuTTY.** In a browser go to https://www.putty.org and get the 64-bit MSI installer. Alternatively, in PowerShell run `winget install PuTTY.PuTTY`.

**Find the COM port.** Switch the board on, then go to Device Manager → **Ports (COM & LPT)** → "USB Serial Port (COMx)". If there are two, use the higher number.
- No port at all usually means a charge-only cable. Try another one.
- A yellow-warning device means the driver is missing. Install the FTDI **VCP** driver from ftdichip.com.

**Connect with PuTTY.** Set Connection type **Serial**, the COM port, Speed **115200**, and Connection → Serial → Flow control **None**. Click Open and press Enter.
- Blank window: press the board's reset button. If still blank, try the other COM port.
- Garbage characters: the speed isn't 115200.

**Log in** as `xilinx` / `xilinx`, then run the checks below and copy the output to the FPGA chat. Selecting text in PuTTY copies it.
```bash
sudo -i                      # password: xilinx
python3 -c "import pynq; print(pynq.__version__)"
lsblk
df -h
python3 -c "from pynq.overlays.base import BaseOverlay; b=BaseOverlay('base.bit'); b.leds[0].on()"
```
The last command should light LED **LD0**. That proves Python can load bitstreams.

**Always shut down before switching off:** run `shutdown -h now`, wait about 10 s, then switch off. Pulling the power can corrupt the card.

---

## Stage 1: build the bitstream (lab PC, ~20–30 min)

1. **Get the code.** Either:
   - `git clone https://github.com/H-Allen/HackTheBurgh.git`, or
   - on GitHub: **Code → Download ZIP**, then unzip.
2. **Open a Vivado shell.**
   - **Windows:** Start menu → **Vivado 20xx.x Tcl Shell**.
   - **Linux:** open a terminal, run `source /<xilinx install>/Vivado/20xx.x/settings64.sh`, then start `vivado -mode tcl`.
   - If several versions are installed, pick the one matching the PYNQ version from Stage 0: PYNQ 3.0.x goes with Vivado 2022.1, and PYNQ 2.7 with 2020.2. Any recent version will probably work.
3. **Run the build** in the Tcl shell, using forward slashes even on Windows:
   ```tcl
   cd C:/path/to/HackTheBurgh/batfpga
   source vivado/build.tcl
   ```
   On Linux you can instead run `cd .../batfpga && vivado -mode batch -source vivado/build.tcl`.
4. **Check the result.** Success ends with:
   ```
   DONE: overlay/bat.bit + overlay/bat.hwh   (FCLK0 = 50 MHz, WNS = <positive> ns)
   ```
   **If it errors**, copy the last ~30 lines (or `batfpga/vivado.log`) to the FPGA chat. The script hasn't been run in Vivado before, so a small fix on the first run is likely.
5. **Bring back the whole `batfpga/overlay/` folder**: `bat.bit`, `bat.hwh` and `reports/`. Use a USB stick, email or Drive. The resource and power numbers for the pitch are in `reports/`.

---

## Stage 2: put the files on the SD card (laptop)

1. **Make a folder called `batfpga`** containing these 8 files:

   | from the repo | file |
   |---|---|
   | `batfpga/overlay/` | `bat.bit`, `bat.hwh` (must be together, same name) |
   | `batfpga/board/` | `run_bat.py` |
   | `batfpga/golden/` | `tree.hex`, `classes.txt`, `audio.wav`, `audio.hex`, `expected_events_fixedfft.txt` |

   On the Mac, `make sdcard` builds this folder in `sdcard/batfpga/`.
2. **Shut the board down properly**, then take the card out and put it in the card reader.
3. **Copy the files.** Windows shows a small drive (the boot partition). Copy the `batfpga` folder onto it.
   **If Windows asks "You need to format the disk…", click CANCEL.** That's the Linux partition, and formatting it erases PYNQ.
4. Eject the card safely, then put it back in the board.

---

## Stage 3: run it on the FPGA (PuTTY)

Boot the board, connect PuTTY, log in, then:
```bash
sudo -i
lsblk                                   # find mmcblk0p1 (the small FAT partition)
mkdir -p /mnt/sd && mount /dev/mmcblk0p1 /mnt/sd     # skip if lsblk already shows a mountpoint
cp -r /mnt/sd/batfpga /home/xilinx/ && cd /home/xilinx/batfpga
ls                                      # should list the 8 files
```

**Test 1: self-test** (real FPGA vs Python, bit-exact):
```bash
python3 run_bat.py audio.hex --expect expected_events_fixedfft.txt
```
It should print `overlay loaded`, then 46 CSV lines like `29,0,244,6,79,30,35,30,79,49,49`, then the stats and finally **`PASS`**. Copy the stats lines (`FPGA work: … cycles/frame … x faster than real time`). Those are your speed numbers.

**Test 2: demo** (calls printed as they happen):
```bash
python3 run_bat.py audio.wav --realtime
```

**Then** run `shutdown -h now` before switching off.

### If something goes wrong
| You see | Meaning / fix |
|---|---|
| `ModuleNotFoundError: No module named 'pynq'` | You forgot `sudo -i`. |
| `bad ID register` | Wrong `.bit`, or `.bit`/`.hwh` from different builds. Copy both again. |
| `AttributeError: ... axi_dma` or `bat_axi_0` | Block names differ. Send the output of `python3 -c "from pynq import Overlay; print(Overlay('bat.bit').ip_dict.keys())"` |
| Hangs with no output | Ctrl-C, then send everything printed. |
| `FAIL` | Send the "first mismatch" line. |

---

## Stage 4: after real data arrives (no lab needed)
Retrain on a laptop, run `make golden`, then copy the new `tree.hex`, `classes.txt` and demo `.wav` files into `batfpga/` on the card. Run Test 2 with the new audio. **The bitstream stays the same.**
