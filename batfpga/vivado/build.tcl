# build.tcl - one-shot Vivado build of the PYNQ-Z2 overlay.
#
#   cd batfpga && vivado -mode batch -source vivado/build.tcl
#
# Produces (copy the first two to the board together, same basename):
#   overlay/bat.bit  overlay/bat.hwh
#   overlay/reports/{utilization,utilization_hier,timing,power}.txt
#
# Block design:  PS7 --GP0--> AXI-Lite: axi_dma (ctrl), bat_axi_0 (regs)
#                PS7 <--HP0-- axi_dma MM2S --AXIS--> bat_axi_0 (audio)
# The decision tree is NOT in the bitstream: run_bat.py loads golden/tree.hex at run time.
# Built successfully on Vivado 2022.2 (bare part): WNS +8.4 ns at 50 MHz, 4.2% LUTs, 7 BRAM, 5 DSP.

set fclk_mhz 50 ;# real time needs ~4.5 MHz; 50 MHz gives ~11x margin and easy timing

set root     [file normalize [file join [file dirname [info script]] ..]]
set proj_dir [file join $root vivado proj]
set out_dir  [file join $root overlay]
set part     xc7z020clg400-1
file mkdir $out_dir [file join $out_dir reports]

create_project -force bat $proj_dir -part $part
set bp [lindex [get_board_parts -quiet -latest_file_version {*pynq-z2*}] 0]
if {$bp ne ""} {
    puts "INFO: using board part $bp"
    set_property board_part $bp [current_project]
} else {
    puts "INFO: PYNQ-Z2 board files not installed; using bare part $part (fine for a PYNQ overlay)"
}

set rtl {bat_axi.v bat_top.v fft256.v fft256_rom.v bat_features.v bat_tree.v tree_walker.v sdp_ram.v}
foreach f $rtl { add_files -norecurse [file join $root rtl $f] }
update_compile_order -fileset sources_1

proc ipv {name} {
    set defs [get_ipdefs -all "xilinx.com:ip:${name}:*"]
    if {[llength $defs] == 0} { error "IP $name not found" }
    return [lindex [lsort -dictionary $defs] end]
}

create_bd_design bat
set ps [create_bd_cell -type ip -vlnv [ipv processing_system7] ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config [list make_external "FIXED_IO, DDR" apply_board_preset [expr {$bp ne "" ? 1 : 0}]] $ps
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $fclk_mhz] $ps

set dma [create_bd_cell -type ip -vlnv [ipv axi_dma] axi_dma]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16}] $dma

create_bd_cell -type module -reference bat_axi bat_axi_0
connect_bd_intf_net [get_bd_intf_pins axi_dma/M_AXIS_MM2S] [get_bd_intf_pins bat_axi_0/s_axis]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins axi_dma/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins bat_axi_0/s_axi]
# (Vivado 2022.2 rejects the short form on the master pin: apply on the HP0 slave with full config)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_dma/M_AXI_MM2S} Slave {/ps7/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} \
    [get_bd_intf_pins ps7/S_AXI_HP0]

# anything automation left unclocked/unreset -> FCLK0 / its reset
foreach p [get_bd_pins -quiet -of_objects [get_bd_cells {axi_dma bat_axi_0}] -filter {TYPE == clk}] {
    if {[llength [get_bd_nets -quiet -of_objects $p]] == 0} { connect_bd_net [get_bd_pins ps7/FCLK_CLK0] $p }
}
set rst [get_bd_cells -quiet -filter {VLNV =~ "*proc_sys_reset*"}]
if {[llength [get_bd_nets -quiet -of_objects [get_bd_pins bat_axi_0/aresetn]]] == 0} {
    connect_bd_net [get_bd_pins [lindex $rst 0]/peripheral_aresetn] [get_bd_pins bat_axi_0/aresetn]
}
assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files bat.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top bat_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed - see $proj_dir" }

open_run impl_1
set rpt [file join $out_dir reports]
report_utilization                -file [file join $rpt utilization.txt]
report_utilization -hierarchical  -file [file join $rpt utilization_hier.txt]
report_timing_summary             -file [file join $rpt timing.txt]
report_power                      -file [file join $rpt power.txt]
report_power -hier all            -file [file join $rpt power_hier.txt]

set bit [glob -nocomplain [file join $proj_dir bat.runs impl_1 *.bit]]
set hwh [lindex [glob -nocomplain \
    [file join $proj_dir bat.gen sources_1 bd bat hw_handoff bat.hwh] \
    [file join $proj_dir bat.srcs sources_1 bd bat hw_handoff bat.hwh]] 0]
file copy -force [lindex $bit 0] [file join $out_dir bat.bit]
file copy -force $hwh            [file join $out_dir bat.hwh]

set wns [get_property STATS.WNS [get_runs impl_1]]
puts "=============================================================="
puts "DONE: overlay/bat.bit + overlay/bat.hwh   (FCLK0 = $fclk_mhz MHz, WNS = $wns ns)"
puts "Reports in overlay/reports/. bat_axi_0 power: see power_hier.txt"
if {$wns < 0} { puts "WARNING: timing NOT met - lower fclk_mhz at the top of this script and rebuild" }
puts "=============================================================="
