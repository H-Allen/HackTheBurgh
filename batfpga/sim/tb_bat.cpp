// tb_bat.cpp - Verilator testbench: streams golden/bins.hex into bat_features,
// logs detected calls, compares against golden/expected_events.txt.
//
// Build & run (from the batfpga/ folder):
//   verilator --cc --exe --build -O2 -Wno-fatal -Irtl --top-module bat_features \
//       rtl/bat_features.v rtl/bat_tree.v sim/tb_bat.cpp -o tb_bat
//   ./obj_dir/tb_bat golden/bins.hex golden/expected_events.txt
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Vbat_features.h"
#include "verilated.h"

struct Ev { long frame; int cls, conf; int f[8]; };

static std::vector<Ev> load_expected(const char* path) {
    std::vector<Ev> v; std::ifstream in(path); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line); Ev e;
        if (!(ss >> e.frame >> e.cls >> e.conf)) continue;
        for (int i = 0; i < 8; i++) ss >> e.f[i];
        v.push_back(e);
    }
    return v;
}

int main(int argc, char** argv) {
    const char* bins_path = argc > 1 ? argv[1] : "golden/bins.hex";
    const char* exp_path  = argc > 2 ? argv[2] : "golden/expected_events.txt";

    VerilatedContext* ctx = new VerilatedContext;
    Vbat_features* top = new Vbat_features{ctx};
    auto tick = [&]() { top->clk = 0; top->eval(); top->clk = 1; top->eval(); };

    top->rst = 1; top->in_valid = 0; top->tree_we = 0; tick(); tick(); top->rst = 0;

    std::vector<Ev> got;
    auto capture = [&]() {
        if (top->out_valid) {
            Ev e; e.frame = top->out_frame; e.cls = top->out_class; e.conf = top->out_conf;
            for (int i = 0; i < 8; i++) e.f[i] = (int)((top->out_feats >> (8 * i)) & 0xff);
            got.push_back(e);
        }
    };

    std::ifstream in(bins_path); std::string line; long n = 0;
    while (std::getline(in, line)) {
        uint32_t w = (uint32_t)std::stoul(line, nullptr, 16);
        top->in_valid = 1; top->in_bin = (w >> 16) & 0x7f; top->in_mag = w & 0xffff;
        tick(); capture(); n++;
    }
    top->in_valid = 0;
    for (int i = 0; i < 4; i++) { tick(); capture(); }

    std::vector<Ev> exp = load_expected(exp_path);
    int bad = 0;
    size_t m = exp.size() > got.size() ? exp.size() : got.size();
    for (size_t i = 0; i < m; i++) {
        bool ok = i < exp.size() && i < got.size() && exp[i].frame == got[i].frame &&
                  exp[i].cls == got[i].cls && exp[i].conf == got[i].conf;
        for (int k = 0; ok && k < 8; k++) ok = exp[i].f[k] == got[i].f[k];
        if (!ok) {
            bad++;
            if (bad <= 10) {
                printf("MISMATCH #%zu  exp: ", i);
                if (i < exp.size()) printf("frame %ld cls %d conf %d", exp[i].frame, exp[i].cls, exp[i].conf); else printf("-");
                printf("   got: ");
                if (i < got.size()) printf("frame %ld cls %d conf %d", got[i].frame, got[i].cls, got[i].conf); else printf("-");
                printf("\n");
            }
        }
    }
    printf("%ld bins (%ld frames) streamed, %zu events expected, %zu detected\n",
           n, n / 128, exp.size(), got.size());
    printf(bad ? "FAIL: %d mismatches\n" : "PASS: RTL matches Python reference bit-exactly\n", bad);

    FILE* f = fopen("sim_events.txt", "w");
    for (auto& e : got) {
        fprintf(f, "%ld %d %d", e.frame, e.cls, e.conf);
        for (int k = 0; k < 8; k++) fprintf(f, " %d", e.f[k]);
        fprintf(f, "\n");
    }
    fclose(f);
    top->final(); delete top; delete ctx;
    return bad ? 1 : 0;
}
