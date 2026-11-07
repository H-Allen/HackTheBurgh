// tb_axi.cpp - system test of the PYNQ wrapper bat_axi.v, using the same register sequence as
// board/run_bat.py: hold core, load tree.hex, release, stream audio in DMA-sized chunks with
// random stalls, drain the event FIFO over AXI-Lite after each chunk, compare with Python.
//
//   make sim_axi
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "Vbat_axi.h"
#include "verilated.h"

static Vbat_axi* top;
static long cycles = 0;
static std::mt19937 rng(5);
static void tick() { top->aclk = 0; top->eval(); top->aclk = 1; top->eval(); cycles++; }

static void axi_write(uint32_t addr, uint32_t data) {
    top->s_axi_awaddr = addr; top->s_axi_wdata = data; top->s_axi_wstrb = 0xf;
    top->s_axi_awvalid = 1; top->s_axi_wvalid = 1; top->s_axi_bready = 1;
    for (int i = 0; i < 100; i++) {
        top->aclk = 0; top->eval();
        bool hs = top->s_axi_awready;           // AW and W handshake at this edge
        tick();
        if (hs) { top->s_axi_awvalid = 0; top->s_axi_wvalid = 0; break; }
    }
    for (int i = 0; i < 100 && !top->s_axi_bvalid; i++) tick();
    tick();                                      // B handshake (bready=1)
    top->s_axi_bready = 0;
}
static uint32_t axi_read(uint32_t addr) {
    top->s_axi_araddr = addr; top->s_axi_arvalid = 1; top->s_axi_rready = 1;
    for (int i = 0; i < 100; i++) {
        top->aclk = 0; top->eval();
        bool hs = top->s_axi_arready;
        tick();
        if (hs) { top->s_axi_arvalid = 0; break; }
    }
    for (int i = 0; i < 100 && !top->s_axi_rvalid; i++) tick();
    uint32_t d = top->s_axi_rdata;
    tick();                                      // R handshake
    top->s_axi_rready = 0;
    return d;
}

enum { CTRL = 0x00, STATUS = 0x04, EV_FRAME = 0x08, EV_CLASS = 0x0C, EV_LO = 0x10, EV_HI = 0x14,
       EV_POP = 0x18, ID = 0x1C, TREE_ADDR = 0x20, TREE_DATA = 0x24, CYC = 0x28, SMP = 0x2C, BUSY = 0x30 };

struct Ev { long frame; int cls, conf; int f[8]; };
static std::vector<uint32_t> load_hex(const char* p) {
    std::vector<uint32_t> v; std::ifstream in(p); std::string l;
    while (std::getline(in, l)) if (!l.empty()) v.push_back((uint32_t)std::stoul(l, nullptr, 16));
    return v;
}

int main(int argc, char** argv) {
    const char* audio_p = argc > 1 ? argv[1] : "golden/audio.hex";
    const char* tree_p  = argc > 2 ? argv[2] : "golden/tree.hex";
    const char* exp_p   = argc > 3 ? argv[3] : "golden/expected_events_fixedfft.txt";
    std::vector<uint32_t> audio = load_hex(audio_p), tree = load_hex(tree_p);
    std::vector<Ev> exp; { std::ifstream in(exp_p); std::string l;
        while (std::getline(in, l)) { std::istringstream s(l); Ev e;
            if (!(s >> e.frame >> e.cls >> e.conf)) continue;
            for (int k = 0; k < 8; k++) s >> e.f[k]; exp.push_back(e); } }

    VerilatedContext* ctx = new VerilatedContext; top = new Vbat_axi{ctx};
    top->aresetn = 0; top->s_axis_tvalid = 0; top->s_axi_awvalid = 0; top->s_axi_wvalid = 0;
    top->s_axi_arvalid = 0; top->s_axi_bready = 0; top->s_axi_rready = 0;
    for (int i = 0; i < 4; i++) tick();
    top->aresetn = 1; tick();

    int bad = 0;
    uint32_t id = axi_read(ID);
    if (id != 0xBA7F0001) { printf("bad ID %08x\n", id); bad++; }
    if (!(axi_read(STATUS) & (1u << 17))) { printf("core not held in reset after aresetn\n"); bad++; }

    axi_write(CTRL, 1);                               // hold core
    axi_write(TREE_ADDR, 0);
    for (uint32_t w : tree) axi_write(TREE_DATA, w);
    axi_write(CTRL, 1 | 2);                           // clear FIFO
    axi_write(CTRL, 0);                               // run

    std::vector<Ev> got;
    auto drain = [&]() {
        uint32_t st = axi_read(STATUS);
        if (st & (1u << 16)) { printf("FIFO overflow\n"); bad++; }
        for (uint32_t n = st & 0x7ff; n; n--) {
            Ev e; e.frame = axi_read(EV_FRAME);
            uint32_t c = axi_read(EV_CLASS); e.cls = (c >> 8) & 0xf; e.conf = c & 0xff;
            uint32_t lo = axi_read(EV_LO), hi = axi_read(EV_HI);
            for (int k = 0; k < 4; k++) { e.f[k] = (lo >> (8 * k)) & 0xff; e.f[k + 4] = (hi >> (8 * k)) & 0xff; }
            axi_write(EV_POP, 1);
            got.push_back(e);
        }
    };

    const size_t CHUNK = 256000;                      // samples per DMA transfer (1 s)
    long stream_cycles = 0;
    for (size_t c0 = 0; c0 < audio.size(); c0 += CHUNK) {
        size_t c1 = std::min(audio.size(), c0 + CHUNK) & ~(size_t)1;
        size_t i = c0; long t0 = cycles;
        while (i < c1) {                              // DMA: 2 samples per word, random stalls
            bool v = (rng() % 8) != 0;
            top->s_axis_tvalid = v;
            top->s_axis_tdata = (audio[i] & 0xffff) | ((audio[i + 1] & 0xffff) << 16);
            top->s_axis_tlast = (i + 2 >= c1);
            top->aclk = 0; top->eval();
            bool hs = v && top->s_axis_tready;
            tick();
            if (hs) i += 2;
        }
        top->s_axis_tvalid = 0;
        for (int k = 0; k < 100000 && !(axi_read(STATUS) & (1u << 18)); k++) ;   // wait idle
        for (int k = 0; k < 64; k++) tick();          // tree walk + FIFO write
        stream_cycles += cycles - t0;
        drain();
    }

    uint32_t smp = axi_read(SMP), busy = axi_read(BUSY), cyc = axi_read(CYC);
    size_t m = std::max(exp.size(), got.size());
    int evbad = 0;
    for (size_t i = 0; i < m; i++) {
        bool ok = i < exp.size() && i < got.size() && exp[i].frame == got[i].frame &&
                  exp[i].cls == got[i].cls && exp[i].conf == got[i].conf;
        for (int k = 0; ok && k < 8; k++) ok = exp[i].f[k] == got[i].f[k];
        if (!ok && ++evbad <= 10) printf("EVENT MISMATCH #%zu\n", i);
    }
    bad += evbad;
    if (smp != (audio.size() & ~(size_t)1)) { printf("SAMPLES %u != %zu\n", smp, audio.size()); bad++; }
    long frames = (long)(smp / 256);
    printf("samples %u, frames %ld, events exp %zu got %zu, mismatches %d\n", smp, frames, exp.size(), got.size(), evbad);
    printf("FFT busy %u cycles = %.0f/frame; core cycles %u; streaming %ld cycles\n",
           busy, (double)busy / frames, cyc, stream_cycles);
    printf(bad ? "FAIL\n" : "PASS: AXI wrapper + programmable tree match Python bit-exactly\n");
    top->final(); delete top; delete ctx;
    return bad ? 1 : 0;
}
