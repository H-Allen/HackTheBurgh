# make sim      - post-FFT RTL vs golden FFT magnitudes (bit-exact)
# make sim_top  - full chain audio -> fft256 -> bat_features (bit-exact vs fft_model, compared to float ref)
# make golden   - regenerate ROMs, bat_tree.v and golden vectors from tree.pkl
PY ?= .venv/bin/python
VFLAGS = --cc --exe --build -O2 -Wno-fatal -Wno-WIDTH -Irtl
POST = rtl/bat_features.v rtl/bat_tree.v
FFT  = rtl/bat_top.v rtl/fft256.v rtl/fft256_rom.v

.PHONY: sim sim_top golden lint clean
sim:
	verilator $(VFLAGS) --top-module bat_features $(POST) sim/tb_bat.cpp -o tb_bat > /dev/null
	./obj_dir/tb_bat golden/bins.hex golden/expected_events.txt
sim_top:
	verilator $(VFLAGS) --top-module bat_top -Mdir obj_dir_top $(FFT) $(POST) sim/tb_top.cpp -o tb_top > /dev/null
	./obj_dir_top/tb_top golden/audio.hex golden/bins_fixed.hex golden/expected_events_fixedfft.txt golden/expected_events.txt
golden:
	$(PY) gen_fft_rom.py && $(PY) export_fpga.py
lint:
	verilator --lint-only -Wall -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME -Irtl --top-module bat_top $(FFT) $(POST)
clean:
	rm -rf obj_dir obj_dir_top sim_events*.txt
