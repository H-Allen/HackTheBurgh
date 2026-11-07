# make sim      - post-FFT RTL vs golden FFT magnitudes (bit-exact)
# make sim_top  - full chain audio -> fft256 -> bat_features (bit-exact vs fft_model, compared to float ref)
# make golden   - regenerate ROMs, bat_tree.v and golden vectors from tree.pkl
PY ?= .venv/bin/python
VFLAGS = --cc --exe --build -O2 -Wno-fatal -Wno-WIDTH -Irtl
POST = rtl/bat_features.v rtl/bat_tree.v rtl/tree_walker.v
FFT  = rtl/bat_top.v rtl/fft256.v rtl/fft256_rom.v

.PHONY: sim sim_top sim_tree sim_axi test golden lint clean sdcard
# make sdcard - collect everything the board needs into sdcard/batfpga/ (copy that folder to the SD card).
# bat.bit/bat.hwh come from the lab Vivado build (overlay/); re-run after retraining to refresh tree.hex.
sdcard:
	mkdir -p sdcard/batfpga
	cp board/run_bat.py golden/tree.hex golden/classes.txt golden/audio.wav golden/audio.hex \
	   golden/expected_events_fixedfft.txt sdcard/batfpga/
	-cp overlay/bat.bit overlay/bat.hwh sdcard/batfpga/ 2>/dev/null || echo "NOTE: no overlay/bat.bit yet - build it in Vivado (vivado/build.tcl)"
	@du -sh sdcard/batfpga
test: sim sim_top sim_tree sim_axi
sim_axi:
	verilator $(VFLAGS) --top-module bat_axi -Mdir obj_dir_axi rtl/bat_axi.v $(FFT) $(POST) sim/tb_axi.cpp -o tb_axi > /dev/null
	./obj_dir_axi/tb_axi golden/audio.hex golden/tree.hex golden/expected_events_fixedfft.txt
sim_tree:
	verilator $(VFLAGS) --top-module tree_eq -Mdir obj_dir_tree sim/tree_eq.v rtl/tree_walker.v rtl/bat_tree.v sim/tb_tree.cpp -o tb_tree > /dev/null
	./obj_dir_tree/tb_tree golden/tree.hex
sim:
	verilator $(VFLAGS) --top-module bat_features $(POST) sim/tb_bat.cpp -o tb_bat > /dev/null
	./obj_dir/tb_bat golden/bins.hex golden/expected_events.txt
sim_top:
	verilator $(VFLAGS) --top-module bat_top -Mdir obj_dir_top $(FFT) $(POST) sim/tb_top.cpp -o tb_top > /dev/null
	./obj_dir_top/tb_top golden/audio.hex golden/bins_fixed.hex golden/expected_events_fixedfft.txt golden/expected_events.txt
golden:
	$(PY) gen_fft_rom.py && $(PY) export_fpga.py
lint:
	verilator --lint-only -Wall -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME -Irtl --top-module bat_axi rtl/bat_axi.v $(FFT) $(POST)
clean:
	rm -rf obj_dir obj_dir_top obj_dir_tree obj_dir_axi sim_events*.txt
