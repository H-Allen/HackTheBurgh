// tb_top.cpp - Verilator testbench for the full chain: audio -> fft256 -> bat_features.
//
// Build & run (from batfpga/):
//   verilator --cc --exe --build -O2 -Wno-fatal -Irtl --top-module bat_top -Mdir obj_dir_top \
//       rtl/bat_top.v rtl/fft256.v rtl/fft256_rom.v rtl/bat_features.v rtl/bat_tree.v \
//       sim/tb_top.cpp -o tb_top
//   ./obj_dir_top/tb_top golden/audio.hex golden/bins_fixed.hex \
//       golden/expected_events_fixedfft.txt golden/expected_events.txt
//
// Checks (fail = exit 1):
//   1. FFT magnitudes == fft_model.fft_mags_fixed()          (bit-exact)
//   2. events          == Python pipeline on fixed-point FFT  (bit-exact)
// Reports (no fail): agreement with the float-FFT reference events (bat_model.fft_mags).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Vbat_top.h"
#include "verilated.h"

struct Ev { long frame; int cls, conf; int f[8]; };

static std::vector<Ev> load_events(const char* path) {
    std::vector<Ev> v; std::ifstream in(path); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line); Ev e;
        if (!(ss >> e.frame >> e.cls >> e.conf)) continue;
        for (int i = 0; i < 8; i++) ss >> e.f[i];
        v.push_back(e);
    }
    return v;
}
static std::vector<uint32_t> load_hex(const char* path) {
    std::vector<uint32_t> v; std::ifstream in(path); std::string line;
    while (std::getline(in, line)) if (!line.empty()) v.push_back((uint32_t)std::stoul(line, nullptr, 16));
    return v;
}
static bool same(const Ev& a, const Ev& b) {
    if (a.frame != b.frame || a.cls != b.cls || a.conf != b.conf) return false;
    for (int k = 0; k < 8; k++) if (a.f[k] != b.f[k]) return false;
    return true;
}

int main(int argc, char** argv) {
    const char* audio_path = argc > 1 ? argv[1] : "golden/audio.hex";
    const char* bins_path  = argc > 2 ? argv[2] : "golden/bins_fixed.hex";
    const char* exp_path   = argc > 3 ? argv[3] : "golden/expected_events_fixedfft.txt";
    const char* ref_path   = argc > 4 ? argv[4] : "golden/expected_events.txt";

    std::vector<uint32_t> audio = load_hex(audio_path);
    std::vector<uint32_t> bins  = load_hex(bins_path);
    std::vector<Ev> exp = load_events(exp_path), ref = load_events(ref_path);

    VerilatedContext* ctx = new VerilatedContext;
    Vbat_top* top = new Vbat_top{ctx};
    top->rst = 1; top->in_valid = 0; top->in_sample = 0; top->tree_we = 0;
    auto tick = [&]() { top->clk = 0; top->eval(); top->clk = 1; top->eval(); };
    tick(); tick(); top->rst = 0;

    std::vector<Ev> got;
    size_t nbin = 0, sent = 0; long bin_bad = 0; long cycles = 0;
    const size_t total_bins = (audio.size() / 256) * 128;
    while (nbin < total_bins && cycles < 100000000L) {
        // drive input (ready/valid); sample presented before the edge, accepted on it
        top->in_valid = sent < audio.size();
        top->in_sample = sent < audio.size() ? (int16_t)(audio[sent] & 0xffff) : 0;
        top->clk = 0; top->eval();
        bool fire = top->in_valid && top->in_ready;
        top->clk = 1; top->eval(); cycles++;
        if (fire) sent++;
        if (top->fft_valid) {
            uint32_t w = ((uint32_t)top->fft_bin << 16) | top->fft_mag;
            if (nbin >= bins.size() || bins[nbin] != w) {
                if (bin_bad < 5) printf("FFT MISMATCH word %zu (frame %zu bin %u): exp %08x got %08x\n",
                                        nbin, nbin / 128, top->fft_bin, nbin < bins.size() ? bins[nbin] : 0, w);
                bin_bad++;
            }
            nbin++;
        }
        if (top->out_valid) {
            Ev e; e.frame = top->out_frame; e.cls = top->out_class; e.conf = top->out_conf;
            for (int i = 0; i < 8; i++) e.f[i] = (int)((top->out_feats >> (8 * i)) & 0xff);
            got.push_back(e);
        }
    }
    top->in_valid = 0;
    for (int i = 0; i < 8; i++) {
        tick();
        if (top->out_valid) {
            Ev e; e.frame = top->out_frame; e.cls = top->out_class; e.conf = top->out_conf;
            for (int k = 0; k < 8; k++) e.f[k] = (int)((top->out_feats >> (8 * k)) & 0xff);
            got.push_back(e);
        }
    }

    // 2. bit-exact events vs fixed-FFT Python pipeline
    int ev_bad = 0;
    size_t m = exp.size() > got.size() ? exp.size() : got.size();
    for (size_t i = 0; i < m; i++) {
        bool ok = i < exp.size() && i < got.size() && same(exp[i], got[i]);
        if (!ok && ++ev_bad <= 10) {
            printf("EVENT MISMATCH #%zu  exp: ", i);
            if (i < exp.size()) printf("frame %ld cls %d conf %d", exp[i].frame, exp[i].cls, exp[i].conf); else printf("-");
            printf("   got: ");
            if (i < got.size()) printf("frame %ld cls %d conf %d", got[i].frame, got[i].cls, got[i].conf); else printf("-");
            printf("\n");
        }
    }

    // 3. closeness to float-FFT reference: event matched if same end frame (+-1) and same class
    int matched = 0, exact = 0;
    std::vector<bool> used(got.size(), false);
    for (auto& r : ref) {
        for (size_t j = 0; j < got.size(); j++) {
            if (used[j] || labs(got[j].frame - r.frame) > 1 || got[j].cls != r.cls) continue;
            used[j] = true; matched++; if (same(got[j], r)) exact++; break;
        }
    }

    long frames = (long)(audio.size() / 256);
    printf("%zu samples (%ld frames), %ld cycles = %.0f cycles/frame -> real time at %.2f MHz\n",
           audio.size(), frames, cycles, (double)cycles / frames, (double)cycles / frames * 1000.0 / 1e6);
    printf("FFT bins: %zu checked, %ld mismatches vs fft_model (bit-exact)\n", nbin, bin_bad);
    printf("events: %zu expected (fixed FFT), %zu detected, %d mismatches\n", exp.size(), got.size(), ev_bad);
    printf("vs float-FFT reference: %d/%zu events same frame(+-1)+class, %d/%zu fully identical, %zu extra\n",
           matched, ref.size(), exact, ref.size(), got.size() - matched);
    bool pass = (bin_bad == 0) && (ev_bad == 0) && nbin == total_bins;
    printf(pass ? "PASS: FFT + post-FFT RTL match fixed-point Python model bit-exactly\n"
                : "FAIL\n");

    FILE* f = fopen("sim_events_top.txt", "w");
    for (auto& e : got) {
        fprintf(f, "%ld %d %d", e.frame, e.cls, e.conf);
        for (int k = 0; k < 8; k++) fprintf(f, " %d", e.f[k]);
        fprintf(f, "\n");
    }
    fclose(f);
    top->final(); delete top; delete ctx;
    return pass ? 0 : 1;
}
