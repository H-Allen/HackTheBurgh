// tree_walker.v - programmable decision tree: walks a node table in RAM, one level per cycle.
// Same decision as the generated bat_tree.v for the same tree, but retraining only needs a new
// table (golden/tree.hex, written by export_fpga.py) loaded at run time - no new bitstream.
//
// Node word (32 bit), table of up to 512 nodes, root at address 0:
//   internal: [31]=0  [30:28] feature  [27:20] threshold  [17:9] right  [8:0] left
//             go left if feats[feature] <= threshold
//   leaf:     [31]=1  [11:8] class  [7:0] confidence
// Feature order = bat_model.FEATURES = byte order of feats: {sweep,bw,max,min,peak,end,start,dur}.
// Latency: start -> done in (depth + 2) cycles. A malformed table (no leaf within MAX_STEPS)
// returns class 15, conf 0 instead of hanging.
`default_nettype none
module tree_walker #(
    parameter integer MAX_STEPS = 16
) (
    input  wire        clk,
    input  wire        rst,
    // table write port
    input  wire        we,
    input  wire [8:0]  waddr,
    input  wire [31:0] wdata,
    // classify
    input  wire        start,          // feats must stay stable until done
    input  wire [63:0] feats,
    output reg         done,
    output reg  [3:0]  cls,
    output reg  [7:0]  conf
);
    reg [31:0] mem [0:511];
    reg [31:0] q;
    reg        walking;
    reg [4:0]  steps;

    wire        is_leaf = q[31];
    wire [2:0]  fidx    = q[30:28];
    wire [7:0]  thr     = q[27:20];
    wire [8:0]  right   = q[17:9];
    wire [8:0]  left    = q[8:0];
    wire [7:0]  fval    = feats[8 * fidx +: 8];
    wire [8:0]  child   = (fval <= thr) ? left : right;
    wire [8:0]  raddr   = start ? 9'd0 : child;

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        q <= mem[raddr];
    end

    always @(posedge clk) begin
        if (rst) begin
            walking <= 1'b0; steps <= 5'd0; done <= 1'b0; cls <= 4'd0; conf <= 8'd0;
        end else begin
            done <= 1'b0;
            if (start) begin
                walking <= 1'b1; steps <= 5'd0;
            end else if (walking) begin
                steps <= steps + 5'd1;
                if (is_leaf) begin
                    walking <= 1'b0; done <= 1'b1; cls <= q[11:8]; conf <= q[7:0];
                end else if (steps == MAX_STEPS[4:0]) begin
                    walking <= 1'b0; done <= 1'b1; cls <= 4'd15; conf <= 8'd0;
                end
            end
        end
    end
endmodule
`default_nettype wire
