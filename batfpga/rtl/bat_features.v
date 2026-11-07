// bat_features.v - post-FFT bat call detector + feature extractor + decision tree.
// Bit-exact with bat_model.py (frame_stats, detect) and bat_tree.v (generated).
//
// Input: FFT magnitude stream, bins 0..127 of each 256-pt frame IN ORDER, one per
//        in_valid cycle (gaps allowed). mag = |re|+|im|, saturated to 16 bits.
// Output: one out_valid pulse per detected call, with features, class id, confidence.
// PROG_TREE=0: fixed tree from bat_tree.v (1-cycle). PROG_TREE=1: tree_walker.v with a node
// table written through tree_we/tree_waddr/tree_wdata (<= 10 cycles; calls are >= 128 cycles apart).
`default_nettype none
module bat_features #(
    parameter integer BIN_LO    = 15,
    parameter integer BIN_HI    = 125,
    parameter integer MIN_E     = 100,
    parameter integer THR_SHIFT = 2,
    parameter integer EMA_SHIFT = 4,
    parameter integer MIN_DUR   = 2,
    parameter integer MAX_DUR   = 63,
    parameter integer PROG_TREE = 0
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_valid,
    input  wire [6:0]  in_bin,
    input  wire [15:0] in_mag,

    input  wire        tree_we,       // PROG_TREE=1 only
    input  wire [8:0]  tree_waddr,
    input  wire [31:0] tree_wdata,

    output reg         out_valid,
    output reg  [31:0] out_frame,     // index of frame where call ended (first quiet frame)
    output reg  [3:0]  out_class,
    output reg  [7:0]  out_conf,
    output reg  [63:0] out_feats      // {sweep,bw,f_max,f_min,f_peak,f_end,f_start,dur} 8b each
);
    localparam [6:0]  LO7   = BIN_LO[6:0];
    localparam [6:0]  HI7   = BIN_HI[6:0];
    localparam [15:0] MINE  = MIN_E[15:0];
    localparam [5:0]  MIND  = MIN_DUR[5:0];
    localparam [5:0]  MAXD  = MAX_DUR[5:0];

    // ---------------- per-frame peak search over band ----------------
    reg [15:0] cur_max;
    reg [6:0]  cur_pk;
    wire in_band = (in_bin >= LO7) && (in_bin <= HI7);

    always @(posedge clk) begin
        if (rst) begin
            cur_max <= 16'd0;
            cur_pk  <= LO7;
        end else if (in_valid) begin
            if (in_bin == 7'd0) begin
                cur_max <= 16'd0;
                cur_pk  <= LO7;
            end else if (in_band && (in_mag > cur_max)) begin
                cur_max <= in_mag;
                cur_pk  <= in_bin;
            end
        end
    end

    wire frame_end = in_valid && (in_bin == 7'd127);
    wire [15:0] P  = cur_max;          // final for this frame (bin 127 is out of band)
    wire [6:0]  pk = cur_pk;

    // ---------------- call detector state ----------------
    reg        ema_init;
    reg [15:0] ema;
    reg        active;
    reg [5:0]  dur;
    reg [6:0]  fs, fe, fpk, fmin, fmax;
    reg [15:0] emax;
    reg [31:0] frame_cnt;

    wire [15:0] ema_cur = ema_init ? ema : P;
    wire [17:0] thr     = {2'b00, ema_cur} << THR_SHIFT;
    wire        act     = ({2'b00, P} > thr) && (P > MINE);

    wire signed [16:0] diff    = $signed({1'b0, P}) - $signed({1'b0, ema_cur});
    wire signed [16:0] diff_sh = diff >>> EMA_SHIFT;
    wire signed [17:0] ema_sum = $signed({2'b00, ema_cur}) + {diff_sh[16], diff_sh};
    wire [15:0] ema_next = ema_sum[15:0];   // EMA of 16-bit values never exceeds 16 bits
    wire        unused_ok = &{1'b0, ema_sum[17:16]};

    // features of the call being closed this frame
    wire [7:0] f_dur   = {2'b00, dur};
    wire [7:0] f_start = {1'b0, fs};
    wire [7:0] f_end   = {1'b0, fe};
    wire [7:0] f_peak  = {1'b0, fpk};
    wire [7:0] f_min   = {1'b0, fmin};
    wire [7:0] f_max   = {1'b0, fmax};
    wire [7:0] f_bw    = {1'b0, fmax - fmin};
    wire [7:0] f_sweep = (fs > fe) ? {1'b0, fs - fe} : 8'd0;

    reg        evt;          // features latched, tree result next cycle
    reg [31:0] evt_frame;
    reg [7:0]  r_dur, r_start, r_end, r_peak, r_min, r_max, r_bw, r_sweep;

    always @(posedge clk) begin
        if (rst) begin
            ema_init <= 1'b0; ema <= 16'd0; active <= 1'b0; dur <= 6'd0;
            fs <= 7'd0; fe <= 7'd0; fpk <= 7'd0; fmin <= 7'd0; fmax <= 7'd0; emax <= 16'd0;
            frame_cnt <= 32'd0; evt <= 1'b0; evt_frame <= 32'd0;
            r_dur <= 8'd0; r_start <= 8'd0; r_end <= 8'd0; r_peak <= 8'd0;
            r_min <= 8'd0; r_max <= 8'd0; r_bw <= 8'd0; r_sweep <= 8'd0;
        end else begin
            evt <= 1'b0;
            if (frame_end) begin
                frame_cnt <= frame_cnt + 32'd1;
                ema_init  <= 1'b1;
                if (act) begin
                    if (!active) begin
                        active <= 1'b1; dur <= 6'd1;
                        fs <= pk; fe <= pk; fmin <= pk; fmax <= pk; fpk <= pk; emax <= P;
                    end else begin
                        if (dur != MAXD) dur <= dur + 6'd1;
                        fe <= pk;
                        if (pk < fmin) fmin <= pk;
                        if (pk > fmax) fmax <= pk;
                        if (P > emax) begin emax <= P; fpk <= pk; end
                    end
                    ema <= ema_cur;
                end else begin
                    if (active) begin
                        active <= 1'b0;
                        if (dur >= MIND) begin
                            evt <= 1'b1; evt_frame <= frame_cnt;
                            r_dur <= f_dur; r_start <= f_start; r_end <= f_end; r_peak <= f_peak;
                            r_min <= f_min; r_max <= f_max; r_bw <= f_bw; r_sweep <= f_sweep;
                        end
                    end
                    ema <= ema_next;
                end
            end
        end
    end

    // ---------------- classifier ----------------
    wire [63:0] r_feats = {r_sweep, r_bw, r_max, r_min, r_peak, r_end, r_start, r_dur};
    wire       t_done;
    wire [3:0] t_class;
    wire [7:0] t_conf;
    generate
        if (PROG_TREE != 0) begin : g_prog
            tree_walker u_walk (
                .clk(clk), .rst(rst),
                .we(tree_we), .waddr(tree_waddr), .wdata(tree_wdata),
                .start(evt), .feats(r_feats),
                .done(t_done), .cls(t_class), .conf(t_conf)
            );
        end else begin : g_fixed
            bat_tree u_tree (
                .f_dur(r_dur), .f_start(r_start), .f_end(r_end), .f_peak(r_peak),
                .f_min(r_min), .f_max(r_max), .f_bw(r_bw), .f_sweep(r_sweep),
                .cls(t_class), .conf(t_conf)
            );
            assign t_done = evt;
            wire unused_tree = &{1'b0, tree_we, tree_waddr, tree_wdata};
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out_frame <= 32'd0; out_class <= 4'd0; out_conf <= 8'd0;
            out_feats <= 64'd0;
        end else begin
            out_valid <= t_done;
            if (t_done) begin
                out_frame <= evt_frame;
                out_class <= t_class;
                out_conf  <= t_conf;
                out_feats <= r_feats;
            end
        end
    end
endmodule
`default_nettype wire
