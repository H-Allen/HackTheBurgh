// tb_tree.cpp - tree_walker (table from golden/tree.hex) == bat_tree.v on 200k feature vectors:
// uniform random + vectors built from the tree's own thresholds (thr, thr+1) to hit every boundary.
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "Vtree_eq.h"
#include "verilated.h"
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "golden/tree.hex";
    std::vector<uint32_t> tab; std::ifstream in(path); std::string l;
    while (std::getline(in, l)) if (!l.empty()) tab.push_back((uint32_t)std::stoul(l, nullptr, 16));
    std::vector<int> thr[8];
    for (uint32_t w : tab) if (!(w >> 31)) { int t = (w >> 20) & 0xff, f = (w >> 28) & 7;
        thr[f].push_back(t); thr[f].push_back(t + 1 > 255 ? 255 : t + 1); }
    for (int f = 0; f < 8; f++) if (thr[f].empty()) thr[f].push_back(0);

    VerilatedContext* ctx = new VerilatedContext; Vtree_eq* t = new Vtree_eq{ctx};
    auto tick = [&]() { t->clk = 0; t->eval(); t->clk = 1; t->eval(); };
    t->rst = 1; t->we = 0; t->start = 0; tick(); t->rst = 0;
    for (size_t a = 0; a < tab.size(); a++) { t->we = 1; t->waddr = a; t->wdata = tab[a]; tick(); }
    t->we = 0;
    std::mt19937_64 rng(1); long bad = 0, n = 200000, maxlat = 0;
    for (long i = 0; i < n; i++) {
        uint64_t f = 0;
        for (int k = 0; k < 8; k++) {
            int v = (i & 1) ? (int)(rng() & 0xff) : thr[k][rng() % thr[k].size()];
            f |= (uint64_t)v << (8 * k);
        }
        t->feats = f; t->start = 1; tick(); t->start = 0;
        long lat = 1; while (!t->done && lat < 40) { tick(); lat++; }
        if (lat > maxlat) maxlat = lat;
        if (!t->done || t->w_cls != t->f_cls || t->w_conf != t->f_conf) {
            if (bad++ < 5) printf("MISMATCH feats %016llx walker %d/%d fixed %d/%d done %d\n",
                (unsigned long long)f, t->w_cls, t->w_conf, t->f_cls, t->f_conf, t->done);
        }
    }
    printf("%ld vectors, %ld mismatches, max latency %ld cycles\n", n, bad, maxlat);
    printf(bad ? "FAIL\n" : "PASS: programmable tree == generated bat_tree.v\n");
    delete t; delete ctx; return bad ? 1 : 0;
}
