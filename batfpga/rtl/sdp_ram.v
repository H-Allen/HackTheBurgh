// sdp_ram.v - simple dual-port RAM, one clock: one write port, one registered read port.
// Written to match Vivado's UG901 "rams_sdp_one_clock" template exactly, so it always infers
// block RAM. Read-during-write to the same address returns the OLD data (read-first); none of the
// users of this module depend on that case. All RAMs in the design go through this module.
`default_nettype none
module sdp_ram #(
    parameter integer W  = 32,
    parameter integer AW = 8
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [W-1:0]  wdata,
    input  wire [AW-1:0] raddr,
    output reg  [W-1:0]  rdata
);
    (* ram_style = "block" *) reg [W-1:0] mem [0:(1<<AW)-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end
    always @(posedge clk) begin
        rdata <= mem[raddr];
    end
endmodule
`default_nettype wire
