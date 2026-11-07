// bat_top.v - int16 audio -> fft256 -> bat_features (detector, features, tree).
// fft_* outputs are a debug tap of the magnitude stream (leave unconnected on the board).
`default_nettype none
module bat_top (
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    output wire               in_ready,
    input  wire signed [15:0] in_sample,

    output wire               out_valid,
    output wire        [31:0] out_frame,
    output wire        [3:0]  out_class,
    output wire        [7:0]  out_conf,
    output wire        [63:0] out_feats,

    output wire               fft_valid,
    output wire        [6:0]  fft_bin,
    output wire        [15:0] fft_mag
);
    fft256 u_fft (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_sample(in_sample),
        .out_valid(fft_valid), .out_bin(fft_bin), .out_mag(fft_mag)
    );
    bat_features u_feat (
        .clk(clk), .rst(rst),
        .in_valid(fft_valid), .in_bin(fft_bin), .in_mag(fft_mag),
        .out_valid(out_valid), .out_frame(out_frame), .out_class(out_class),
        .out_conf(out_conf), .out_feats(out_feats)
    );
endmodule
`default_nettype wire
