// fft256.v - low-power iterative 256-pt FFT front end. Bit-exact with fft_model.fft_mags_fixed().
//
// One radix-2 butterfly unit (4 multipliers), one 256 x 44-bit dual-port RAM.
// Per 256-sample frame:  LOAD 256 cycles (Hann window, bit-reversed write)
//                        8 stages x 128 butterflies x 3 cycles = 3072 cycles
//                        OUT 129 cycles (bins 0..127, mag = |re|+|im|)
// ~3460 cycles per 1 ms frame -> real time needs only ~3.5 MHz; at 100 MHz it runs ~29x
// faster than real time and idles the rest (clock-gate / lower the clock for power).
//
// Input: int16 samples with valid/ready (ready is low while computing; feed from BRAM/DDR,
//        or put a >=256-sample FIFO in front for a live 256 kHz stream).
// Output: bins 0..127 of each frame in order, one per out_valid cycle -> bat_features.
`default_nettype none
module fft256 (
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    output wire               in_ready,
    input  wire signed [15:0] in_sample,

    output reg                out_valid,
    output reg         [6:0]  out_bin,
    output reg         [15:0] out_mag
);
    localparam integer DW   = 22;   // fft_model.DW
    localparam integer FRAC = 4;    // fft_model.FRAC

    localparam [2:0] S_LOAD = 3'd0, S_RD = 3'd1, S_MUL = 3'd2, S_WR = 3'd3, S_OUT = 3'd4;
    reg [2:0] state;
    reg [7:0] cnt;                 // LOAD: sample index; OUT: bin index
    reg [2:0] stage;
    reg [6:0] bf;                  // butterfly index within stage

    // ---------------- dual-port RAM {re, im} ----------------
    reg  [2*DW-1:0] mem [0:255];
    reg  [7:0]      addr_a, addr_b;
    reg             we_a, we_b;
    reg  [2*DW-1:0] din_a, din_b;
    reg  [2*DW-1:0] dout_a, dout_b;
    always @(posedge clk) begin
        if (we_a) mem[addr_a] <= din_a;
        dout_a <= mem[addr_a];
    end
    always @(posedge clk) begin
        if (we_b) mem[addr_b] <= din_b;
        dout_b <= mem[addr_b];
    end

    // ---------------- load: window + bit-reverse ----------------
    wire signed [15:0] hann;
    hann_rom u_hann (.addr(cnt), .q(hann));
    wire signed [31:0] xprod = in_sample * hann;
    wire signed [15:0] xw    = xprod[30:15];                     // (x*w) >> 15, fits int16
    wire signed [DW-1:0] xw_ext = {{(DW-16-FRAC){xw[15]}}, xw, {FRAC{1'b0}}};
    wire [7:0] cnt_rev = {cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7]};
    assign in_ready = (state == S_LOAD);

    // ---------------- butterfly addressing ----------------
    wire [7:0] bf8  = {1'b0, bf};
    wire [7:0] hmsk = (8'd1 << stage) - 8'd1;                   // half - 1
    wire [7:0] i1   = ((bf8 >> stage) << (stage + 3'd1)) | (bf8 & hmsk);
    wire [7:0] i2   = i1 | (8'd1 << stage);
    wire [6:0] twk  = bf & hmsk[6:0];
    wire [6:0] twa  = twk << (3'd7 - stage);
    wire signed [17:0] tw_c, tw_s;
    twiddle_rom u_tw (.addr(twa), .c(tw_c), .s(tw_s));

    // ---------------- butterfly datapath ----------------
    reg  signed [17:0]   c_r, s_r;
    reg  signed [DW-1:0] ar, ai;
    reg  signed [39:0]   p_rc, p_is, p_rs, p_ic;
    wire signed [DW-1:0] br = dout_b[2*DW-1:DW];
    wire signed [DW-1:0] bi = dout_b[DW-1:0];

    wire signed [40:0] tr_full = $signed({p_rc[39], p_rc}) - $signed({p_is[39], p_is}) + 41'sd32768;
    wire signed [40:0] ti_full = $signed({p_rs[39], p_rs}) + $signed({p_ic[39], p_ic}) + 41'sd32768;
    wire signed [24:0] tr = tr_full[40:16];                      // >>> 16 (|t| < 2^24)
    wire signed [24:0] ti = ti_full[40:16];
    wire signed [25:0] sr = {{4{ar[DW-1]}}, ar} + {tr[24], tr} + 26'sd1;
    wire signed [25:0] si = {{4{ai[DW-1]}}, ai} + {ti[24], ti} + 26'sd1;
    wire signed [25:0] dr = {{4{ar[DW-1]}}, ar} - {tr[24], tr} + 26'sd1;
    wire signed [25:0] di = {{4{ai[DW-1]}}, ai} - {ti[24], ti} + 26'sd1;

    // ---------------- output: round off guard bits, |re|+|im| ----------------
    wire signed [DW-1:0] ore = dout_a[2*DW-1:DW];
    wire signed [DW-1:0] oim = dout_a[DW-1:0];
    wire signed [DW:0]   ore_r = {ore[DW-1], ore} + (1 << (FRAC - 1));
    wire signed [DW:0]   oim_r = {oim[DW-1], oim} + (1 << (FRAC - 1));
    wire signed [DW-FRAC:0] ore_s = ore_r[DW:FRAC];
    wire signed [DW-FRAC:0] oim_s = oim_r[DW:FRAC];
    wire [DW-FRAC:0] ore_a = ore_s[DW-FRAC] ? -ore_s : ore_s;
    wire [DW-FRAC:0] oim_a = oim_s[DW-FRAC] ? -oim_s : oim_s;
    wire [DW-FRAC+1:0] osum = {1'b0, ore_a} + {1'b0, oim_a};
    wire [15:0] omag = (osum > 65535) ? 16'hffff : osum[15:0];
    reg        out_pend;           // RAM read issued last cycle for bin out_pend_bin
    reg  [6:0] out_pend_bin;

    always @(*) begin
        addr_a = 8'd0; addr_b = 8'd0; we_a = 1'b0; we_b = 1'b0;
        din_a = {2*DW{1'b0}}; din_b = {2*DW{1'b0}};
        case (state)
            S_LOAD: begin
                addr_a = cnt_rev; din_a = {xw_ext, {DW{1'b0}}};
                we_a = in_valid;
            end
            S_RD, S_MUL: begin addr_a = i1; addr_b = i2; end
            S_WR: begin
                addr_a = i1; addr_b = i2; we_a = 1'b1; we_b = 1'b1;
                din_a = {sr[DW:1], si[DW:1]};
                din_b = {dr[DW:1], di[DW:1]};
            end
            S_OUT: addr_a = cnt;
            default: ;
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_LOAD; cnt <= 8'd0; stage <= 3'd0; bf <= 7'd0;
            out_valid <= 1'b0; out_bin <= 7'd0; out_mag <= 16'd0;
            out_pend <= 1'b0; out_pend_bin <= 7'd0;
            c_r <= 18'sd0; s_r <= 18'sd0; ar <= 0; ai <= 0;
            p_rc <= 40'sd0; p_is <= 40'sd0; p_rs <= 40'sd0; p_ic <= 40'sd0;
        end else begin
            out_valid <= out_pend;
            out_bin   <= out_pend_bin;
            out_mag   <= omag;
            out_pend  <= 1'b0;
            case (state)
                S_LOAD: if (in_valid) begin
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd255) begin state <= S_RD; stage <= 3'd0; bf <= 7'd0; end
                end
                S_RD: begin
                    c_r <= tw_c; s_r <= tw_s;
                    state <= S_MUL;
                end
                S_MUL: begin
                    ar <= dout_a[2*DW-1:DW]; ai <= dout_a[DW-1:0];
                    p_rc <= br * c_r; p_is <= bi * s_r;
                    p_rs <= br * s_r; p_ic <= bi * c_r;
                    state <= S_WR;
                end
                S_WR: begin
                    bf <= bf + 7'd1;
                    state <= S_RD;
                    if (bf == 7'd127) begin
                        stage <= stage + 3'd1;
                        if (stage == 3'd7) begin state <= S_OUT; cnt <= 8'd0; end
                    end
                end
                S_OUT: begin
                    out_pend <= 1'b1; out_pend_bin <= cnt[6:0];
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd127) begin state <= S_LOAD; cnt <= 8'd0; end
                end
                default: state <= S_LOAD;
            endcase
        end
    end
endmodule
`default_nettype wire
