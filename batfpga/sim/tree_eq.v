// tree_eq.v - test harness: programmable tree_walker vs generated bat_tree on the same features.
`default_nettype none
module tree_eq (
    input  wire        clk, rst, we, start,
    input  wire [8:0]  waddr,
    input  wire [31:0] wdata,
    input  wire [63:0] feats,
    output wire        done,
    output wire [3:0]  w_cls, f_cls,
    output wire [7:0]  w_conf, f_conf
);
    tree_walker u_w (.clk(clk), .rst(rst), .we(we), .waddr(waddr), .wdata(wdata),
                     .start(start), .feats(feats), .done(done), .cls(w_cls), .conf(w_conf));
    bat_tree u_f (.f_dur(feats[7:0]), .f_start(feats[15:8]), .f_end(feats[23:16]),
                  .f_peak(feats[31:24]), .f_min(feats[39:32]), .f_max(feats[47:40]),
                  .f_bw(feats[55:48]), .f_sweep(feats[63:56]), .cls(f_cls), .conf(f_conf));
endmodule
`default_nettype wire
