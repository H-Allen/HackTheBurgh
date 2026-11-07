#!/usr/bin/env python3
"""
serial_reader.py - laptop side: read the board's serial console and keep only the CSV call lines.

    pip install pyserial
    python serial_reader.py /dev/tty.usbserial-XXXX1 calls.csv      # Mac (Linux: /dev/ttyUSB1, Win: COM5)

Appends each line `frame_idx,class_id,confidence,dur,f_start,f_end,f_peak,f_min,f_max,bw,sweep`
to calls.csv (and stdout) as it arrives, so the dashboard can tail the file.
Shell prompts, logs and species lines on the console are ignored.
Note: close `screen` first - only one program can hold the serial port. Start run_bat.py from
this script with --cmd, or type into screen, exit it (Ctrl-A K), then start this reader.
"""
import argparse, re, sys
import serial

CSV = re.compile(r"^\d+(,\d+){10}$")

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("out", nargs="?", default="calls.csv")
ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--cmd", help='command to send first, e.g. "python3 run_bat.py audio.wav --realtime --quiet"')
a = ap.parse_args()

s = serial.Serial(a.port, a.baud, timeout=1)
if a.cmd:
    s.write((a.cmd + "\n").encode())
with open(a.out, "a") as f:
    while True:
        line = s.readline().decode(errors="ignore").strip()
        if CSV.match(line):
            print(line, flush=True)
            f.write(line + "\n"); f.flush()
