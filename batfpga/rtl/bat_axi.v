// bat_axi.v - PYNQ wrapper around bat_top (PROG_TREE=1).
//
//   PS DDR --AXI DMA (MM2S)--> s_axis: 32-bit words, 2 int16 samples each (low half first),
//                              i.e. a numpy int16 array copied straight into the DMA buffer.
//   PS GP0 --AXI-Lite-------> s_axi: control, tree table, event FIFO, counters.
//
// Register map (byte offsets):
//   0x00 CTRL       RW  [0] core_rst (1 = hold detector/FFT in reset; reset value 1)
//                       [1] write 1: clear event FIFO + overflow flag
//   0x04 STATUS     RO  [10:0] events in FIFO, [16] FIFO overflowed, [17] core_rst, [18] idle
//   0x08 EV_FRAME   RO  head event: frame index (first quiet frame after the call)
//   0x0C EV_CLASS   RO  head event: [11:8] class id, [7:0] confidence
//   0x10 EV_FEAT_LO RO  head event: {f_peak, f_end, f_start, dur}      (byte 0 = dur)
//   0x14 EV_FEAT_HI RO  head event: {sweep, bw, f_max, f_min}
//   0x18 EV_POP     WO  any write: drop head event
//   0x1C ID         RO  0xBA7F_0001
//   0x20 TREE_ADDR  RW  next tree table address (0..511)
//   0x24 TREE_DATA  WO  write node word to tree[TREE_ADDR], TREE_ADDR += 1
//   0x28 CYCLES     RO  clock cycles while core out of reset
//   0x2C SAMPLES    RO  audio samples consumed while core out of reset
//   0x30 BUSY       RO  cycles the FFT was computing (not waiting for samples)
`default_nettype none
module bat_axi (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    input  wire [7:0]  s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [7:0]  s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready
);
    localparam integer FAW = 10;                 // event FIFO depth 1024
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    reg  core_rst_r;
    wire core_rst = core_rst_r | ~aresetn;

    // ---------------- stream unpacker: 1 word -> 2 samples ----------------
    reg  [31:0] word;
    reg  [1:0]  rem;                             // samples left in word
    wire        core_ready;
    wire        core_valid = (rem != 2'd0);
    wire        core_fire  = core_valid & core_ready;
    wire signed [15:0] core_sample = (rem == 2'd2) ? word[15:0] : word[31:16];
    assign s_axis_tready = ~core_rst & ((rem == 2'd0) | ((rem == 2'd1) & core_fire));
    always @(posedge aclk) begin
        if (core_rst) rem <= 2'd0;
        else if (s_axis_tvalid & s_axis_tready) begin word <= s_axis_tdata; rem <= 2'd2; end
        else if (core_fire) rem <= rem - 2'd1;
    end

    // ---------------- core ----------------
    reg         tree_we;
    reg  [8:0]  tree_addr;
    reg  [8:0]  tree_waddr;
    reg  [31:0] tree_wdata;
    wire        ev_valid, fft_busy;
    wire [31:0] ev_frame;
    wire [3:0]  ev_class;
    wire [7:0]  ev_conf;
    wire [63:0] ev_feats;
    wire        fft_valid_unused;
    wire [6:0]  fft_bin_unused;
    wire [15:0] fft_mag_unused;
    bat_top #(.PROG_TREE(1)) u_core (
        .clk(aclk), .rst(core_rst),
        .in_valid(core_valid), .in_ready(core_ready), .in_sample(core_sample),
        .tree_we(tree_we), .tree_waddr(tree_waddr), .tree_wdata(tree_wdata),
        .fft_busy(fft_busy),
        .out_valid(ev_valid), .out_frame(ev_frame), .out_class(ev_class),
        .out_conf(ev_conf), .out_feats(ev_feats),
        .fft_valid(fft_valid_unused), .fft_bin(fft_bin_unused), .fft_mag(fft_mag_unused)
    );

    // ---------------- counters ----------------
    reg [31:0] cyc_cnt, smp_cnt, busy_cnt;
    always @(posedge aclk) begin
        if (core_rst) begin
            cyc_cnt <= 32'd0; smp_cnt <= 32'd0; busy_cnt <= 32'd0;
        end else begin
            cyc_cnt <= cyc_cnt + 32'd1;
            if (core_fire) smp_cnt  <= smp_cnt + 32'd1;
            if (fft_busy)  busy_cnt <= busy_cnt + 32'd1;
        end
    end

    // ---------------- event FIFO (BRAM, head register always shows mem[rd]) ----------------
    reg  [FAW:0]  wp, rp;
    wire [107:0]  head;
    reg           ovf;
    reg           pop, fclr;
    wire [FAW:0]  fcount = wp - rp;
    wire          fempty = (fcount == 0);
    wire          ffull  = (fcount == (1 << FAW));
    wire [FAW:0]  rp_next = fclr ? wp : (pop & ~fempty) ? rp + 1'b1 : rp;
    sdp_ram #(.W(108), .AW(FAW)) u_fifo (
        .clk(aclk), .we(ev_valid & ~ffull), .waddr(wp[FAW-1:0]),
        .wdata({ev_frame, ev_class, ev_conf, ev_feats}),
        .raddr(rp_next[FAW-1:0]), .rdata(head));
    always @(posedge aclk) begin
        if (~aresetn) begin
            wp <= 0; rp <= 0; ovf <= 1'b0;
        end else begin
            rp <= rp_next;
            if (ev_valid) begin
                if (~ffull) wp <= wp + 1'b1;
                else        ovf <= 1'b1;
            end
            if (fclr) ovf <= 1'b0;
        end
    end

    wire idle = ~fft_busy & (rem == 2'd0);

    // ---------------- AXI-Lite ----------------
    wire [5:0] waddr_w = s_axi_awaddr[7:2];
    wire [5:0] raddr_w = s_axi_araddr[7:2];
    always @(posedge aclk) begin
        if (~aresetn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
            s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0;
            core_rst_r <= 1'b1; tree_addr <= 9'd0; tree_we <= 1'b0;
            tree_waddr <= 9'd0; tree_wdata <= 32'd0; pop <= 1'b0; fclr <= 1'b0;
        end else begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_arready <= 1'b0;
            tree_we <= 1'b0; pop <= 1'b0; fclr <= 1'b0;

            // write: accept AW+W together, one transaction at a time
            if (~s_axi_awready & s_axi_awvalid & s_axi_wvalid & ~s_axi_bvalid) begin
                s_axi_awready <= 1'b1; s_axi_wready <= 1'b1;
            end
            if (s_axi_awready) begin             // handshake completes this cycle
                s_axi_bvalid <= 1'b1;
                case (waddr_w)
                    6'h00: begin core_rst_r <= s_axi_wdata[0]; fclr <= s_axi_wdata[1]; end
                    6'h06: pop <= 1'b1;
                    6'h08: tree_addr <= s_axi_wdata[8:0];
                    6'h09: begin
                        tree_we <= 1'b1; tree_waddr <= tree_addr; tree_wdata <= s_axi_wdata;
                        tree_addr <= tree_addr + 9'd1;
                    end
                    default: ;
                endcase
            end else if (s_axi_bvalid & s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            // read
            if (~s_axi_arready & s_axi_arvalid & ~s_axi_rvalid) s_axi_arready <= 1'b1;
            if (s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (raddr_w)
                    6'h00: s_axi_rdata <= {31'd0, core_rst_r};
                    6'h01: s_axi_rdata <= {13'd0, idle, core_rst, ovf, 5'd0, fcount};
                    6'h02: s_axi_rdata <= head[107:76];
                    6'h03: s_axi_rdata <= {20'd0, head[75:72], head[71:64]};
                    6'h04: s_axi_rdata <= head[31:0];
                    6'h05: s_axi_rdata <= head[63:32];
                    6'h07: s_axi_rdata <= 32'hBA7F_0001;
                    6'h08: s_axi_rdata <= {23'd0, tree_addr};
                    6'h0A: s_axi_rdata <= cyc_cnt;
                    6'h0B: s_axi_rdata <= smp_cnt;
                    6'h0C: s_axi_rdata <= busy_cnt;
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid & s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    wire unused_ok = &{1'b0, s_axis_tlast, s_axi_awprot, s_axi_arprot, s_axi_wstrb,
                       s_axi_awaddr[1:0], s_axi_araddr[1:0],
                       fft_valid_unused, fft_bin_unused, fft_mag_unused};
endmodule
`default_nettype wire
