"""
network_sim.py - "what we know now" vs "what a network of FPGA bat sensors would let us predict".

An observing-system simulation (OSSE), the standard way to evaluate a monitoring network before
building it:

  1. REAL DATA - ChirosetEurope records in the region: how much is known today, where, when.
  2. TEST WORLD - a plausible map of the share of bat activity that comes from SARS-related-
     coronavirus reservoirs (horseshoe bats, Rhinolophus): a smooth surface estimated from the real
     records, plus roost hotspots - half at real horseshoe-bat record sites ("known roosts"), half
     in places nobody has recorded ("undiscovered roosts"). Because we know the truth in this world,
     any survey strategy can be scored against it.
  3. TWO WAYS OF LEARNING ABOUT IT
       today:   the REAL locations of existing recordings, each revealing one bat pass (expert ID)
       sensors: N FPGA sensors spread over the region, listening for some nights; every pass is
                classified with our classifier's MEASURED error rates (held-out real data)
  4. ONE PREDICTION MODEL for all scenarios - a Gaussian-process spatial model (kriging) of the
     reservoir share, which gives a predicted map AND its uncertainty everywhere.
  5. SCORE against the test world: hotspots found (especially undiscovered ones), area where we
     cannot tell, map error - as a function of the number of sensors and nights.

  python predict/network_sim.py            -> predict/figures/*.png, predict/results.json
The region is a parameter (REGION); the method transfers to e.g. South-East Asia given records.
Presence-only citizen-science data: the map is "share of recorded bat activity", not abundance.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "figures")

REGION = dict(name="Greece & Bulgaria", countries=["Greece", "Bulgaria"],
              lat=(34.8, 44.3), lon=(19.3, 28.7))
GRID = 0.1                         # deg (~10 km cells)
RESERVOIR = "Rhinolophus"          # SARS-related CoV reservoirs (horseshoe bats)
# classifier error rates, "horseshoe vs not", per pass, from held-out real data (validation split,
# the more conservative of val/test): 3888/3898 and 1609/1615
SENS, SPEC = 3888 / 3898, 1609 / 1615
PASSES_PER_NIGHT = 40              # assumption: bat passes a sensor classifies per night
ELL_KM, PRIOR_VAR, NUGGET = 12.0, 1.5, 0.05     # GP: length scale ~ roost foraging radius, prior var, nugget
N_HOT = 30                         # roost hotspots in the test world (half undiscovered)
HIGH = 0.4                         # "high risk": >40% of bat activity from horseshoe bats
SEEDS = 8
KM = 111.0
COSLAT = np.cos(np.radians(np.mean(REGION["lat"])))


def xy(lat, lon):
    return np.c_[np.asarray(lon) * KM * COSLAT, np.asarray(lat) * KM]


def logit(p):
    return np.log(p / (1 - p))


def sigm(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ geography
def load_land():
    gj = json.load(open(os.path.join(HERE, "data", "ne_50m_countries.geojson")))
    region_paths, outlines = [], []
    for f in gj["features"]:
        g, name = f["geometry"], f["properties"]["ADMIN"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            if (ring[:, 0].max() < REGION["lon"][0] - 3 or ring[:, 0].min() > REGION["lon"][1] + 3 or
                    ring[:, 1].max() < REGION["lat"][0] - 3 or ring[:, 1].min() > REGION["lat"][1] + 3):
                continue
            outlines.append(ring)
            if name in REGION["countries"]:
                region_paths.append(Path(ring))
    lats = np.arange(REGION["lat"][0], REGION["lat"][1], GRID) + GRID / 2
    lons = np.arange(REGION["lon"][0], REGION["lon"][1], GRID) + GRID / 2
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.c_[LON.ravel(), LAT.ravel()]
    land = np.zeros(len(pts), bool)
    for p in region_paths:
        land |= p.contains_points(pts)
    return lats, lons, LAT, LON, land.reshape(LAT.shape), outlines


# ------------------------------------------------------------------ GP (kriging) on logit share
def matern32(d):
    a = np.sqrt(3) * d / ELL_KM
    return PRIOR_VAR * (1 + a) * np.exp(-a)


def site_obs(locs, k, n, sens=1.0, spec=1.0):
    """per-site logit estimate of the TRUE reservoir share and its variance, corrected for
    classifier errors (Rogan-Gladen + delta method)."""
    j = sens + spec - 1
    p_obs = (k + 0.5) / (n + 1)
    p = np.clip((p_obs - (1 - spec)) / j, 0.01, 0.99)
    var_p = p_obs * (1 - p_obs) / (n + 1) / j ** 2
    return locs, logit(p), var_p / (p * (1 - p)) ** 2 + NUGGET


def gp_predict(obs_xy, y, var, grid_xy, mu0):
    if len(y) == 0:
        return np.full(len(grid_xy), mu0), np.full(len(grid_xy), np.sqrt(PRIOR_VAR))
    d = np.linalg.norm(obs_xy[:, None] - obs_xy[None], axis=2)
    K = matern32(d) + np.diag(var)
    L = np.linalg.cholesky(K)
    Ks = matern32(np.linalg.norm(grid_xy[:, None] - obs_xy[None], axis=2))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu0))
    v = np.linalg.solve(L, Ks.T)
    return mu0 + Ks @ alpha, np.sqrt(np.maximum(PRIOR_VAR - (v * v).sum(0), 1e-9))


def aggregate(locs_ll, labels):
    """records -> unique sites (0.01 deg): xy, k reservoir, n total"""
    df = pd.DataFrame(dict(lat=np.round(locs_ll[:, 0], 2), lon=np.round(locs_ll[:, 1], 2), r=labels))
    g = df.groupby(["lat", "lon"]).r.agg(["sum", "count"]).reset_index()
    return xy(g.lat, g.lon), g["sum"].values.astype(float), g["count"].values.astype(float)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    lats, lons, LAT, LON, land, outlines = load_land()
    gxy = xy(LAT[land], LON[land])
    n_land = land.sum()
    cell_km2 = (GRID * KM) * (GRID * KM * COSLAT)

    # ---------------- 1. what we know now (real records) ----------------
    m = pd.read_csv(os.path.join(ROOT, "ChirosetEurope_manifest_v1.csv"), low_memory=False)
    m = m[m.country.isin(REGION["countries"]) & m.latitude.between(*REGION["lat"])
          & m.longitude.between(*REGION["lon"])].copy()
    m["res"] = m.latin.str.startswith(RESERVOIR)
    m["date"] = pd.to_datetime(m.date, errors="coerce")
    rec_ll = m[["latitude", "longitude"]].values
    ci = ((m.latitude - REGION["lat"][0]) // GRID).astype(int).clip(0, len(lats) - 1)
    cj = ((m.longitude - REGION["lon"][0]) // GRID).astype(int).clip(0, len(lons) - 1)
    cells_any = set(zip(ci, cj)) & set(zip(*np.where(land)))
    cells_res = set(zip(ci[m.res], cj[m.res])) & set(zip(*np.where(land)))
    months = m.date.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    yrs = m.date.dt.year[m.date.dt.year >= 1950]
    known = dict(
        region=REGION["name"], records=int(len(m)), species=int(m.latin.nunique()),
        reservoir_records=int(m.res.sum()), reservoir_species=int(m[m.res].latin.nunique()),
        sites=int(m[["latitude", "longitude"]].round(2).drop_duplicates().shape[0]),
        land_cells=int(n_land), cell_km2=round(cell_km2, 1), land_area_km2=int(n_land * cell_km2),
        cells_with_any_record_pct=round(100 * len(cells_any) / n_land, 1),
        cells_with_reservoir_record_pct=round(100 * len(cells_res) / n_land, 1),
        records_jun_aug_pct=round(100 * months[[6, 7, 8]].sum() / months.sum(), 1),
        records_nov_feb=int(months[[11, 12, 1, 2]].sum()),
        years=f"{int(yrs.min())}-{int(yrs.max())}", records_per_month=months.to_dict())
    print("WHAT WE KNOW NOW:", json.dumps(known, indent=1))

    # ---------------- 2. test-world base surface from the real records ----------------
    rxy = xy(rec_ll[:, 0], rec_ll[:, 1])
    mean_share = m.res.mean()
    w = np.exp(-0.5 * (np.linalg.norm(gxy[:, None] - rxy[None], axis=2) / 30.0) ** 2)
    base = np.clip((w @ m.res.values + 2 * mean_share) / (w.sum(1) + 2), 0.02, 0.9)
    dist_to_rec = np.linalg.norm(gxy[:, None] - rxy[None], axis=2).min(1)
    res_sites = np.unique(np.round(rec_ll[m.res.values], 2), axis=0)
    mu0 = logit(mean_share)

    def make_world(rng):
        known_h = res_sites[rng.choice(len(res_sites), N_HOT // 2, replace=False)]
        far = np.where(dist_to_rec > 25)[0]
        unk_idx = rng.choice(far, N_HOT - N_HOT // 2, replace=False)
        hot_xy = np.r_[xy(known_h[:, 0], known_h[:, 1]), gxy[unk_idx]]
        z = logit(base)
        hot_cell = np.array([np.argmin(np.linalg.norm(gxy - h, axis=1)) for h in hot_xy])
        sig = rng.uniform(6, 12, N_HOT)                          # roost foraging radius (km)
        # each roost lifts the local horseshoe share to at least ~0.65-0.9
        amp = np.maximum(rng.uniform(2.0, 3.0, N_HOT), logit(rng.uniform(0.65, 0.9, N_HOT)) - z[hot_cell])
        for h, a, s in zip(hot_xy, amp, sig):
            z = z + a * np.exp(-0.5 * (np.linalg.norm(gxy - h, axis=1) / s) ** 2)
        truth = sigm(z)
        undiscovered = np.r_[np.zeros(N_HOT // 2, bool), np.ones(N_HOT - N_HOT // 2, bool)]
        return truth, hot_xy, hot_cell, undiscovered

    def truth_at(truth, pts_xy):
        return truth[np.argmin(np.linalg.norm(gxy[None] - pts_xy[:, None], axis=2), axis=1)]

    def sensor_sites(n, rng):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.integers(1e9))).fit(gxy)
        return gxy[[np.argmin(np.linalg.norm(gxy - c, axis=1)) for c in km.cluster_centers_]]

    def evaluate(truth, hot_cell, undisc, mean, sd):
        p_high = 1 - norm.cdf((logit(HIGH) - mean) / sd)         # P(share > HIGH)
        flagged = p_high > 0.7                                    # confidently high risk
        found = flagged[hot_cell]
        high = truth > HIGH
        return dict(
            hot_found=float(found.mean()), undiscovered_found=float(found[undisc].mean()),
            known_found=float(found[~undisc].mean()),
            high_area_flagged_pct=float(100 * flagged[high].mean()) if high.any() else 0.0,
            unknown_area_pct=float(100 * (sd > 1.0).mean()),
            false_alarm_area_pct=float(100 * (flagged & (truth < 0.2)).mean()),
            rmse=float(np.sqrt(np.mean((sigm(mean) - truth) ** 2))))

    def run(rng, truth, hot_cell, undisc, n_sensors, nights, sens, spec, with_current=True, keep=False):
        obs = []
        if with_current:                                           # today: real record locations
            lab = rng.random(len(rec_ll)) < truth_at(truth, rxy)
            sx, k, n = aggregate(rec_ll, lab)
            obs.append(site_obs(sx, k, n))
        if n_sensors:
            sxy = sensor_sites(n_sensors, rng)
            r = truth_at(truth, sxy)
            q = sens * r + (1 - spec) * (1 - r)                    # P(sensor says "horseshoe")
            npass = nights * PASSES_PER_NIGHT
            k = rng.binomial(npass, q).astype(float)
            obs.append(site_obs(sxy, k, np.full(n_sensors, float(npass)), sens, spec))
        if obs:
            X = np.vstack([o[0] for o in obs]); y = np.concatenate([o[1] for o in obs])
            v = np.concatenate([o[2] for o in obs])
        else:
            X, y, v = np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        mean, sd = gp_predict(X, y, v, gxy, mu0)
        res = evaluate(truth, hot_cell, undisc, mean, sd)
        if keep:
            res.update(mean=mean, sd=sd, sensors=sxy if n_sensors else None)
        return res

    # ---------------- 3-5. scenarios ----------------
    N_LIST, NIGHTS = [0, 25, 50, 100, 250, 500, 1000], [1, 3, 7, 14, 30]
    N_MAP, N_NIGHTS = 500, 250
    rows = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        truth, hot_xy, hot_cell, undisc = make_world(rng)
        for n in N_LIST:
            for clf, (se, sp) in [("FPGA classifier", (SENS, SPEC)), ("perfect classifier", (1.0, 1.0))]:
                if n == 0 and clf != "FPGA classifier":
                    continue
                crn = np.random.default_rng([seed, n])      # same placement + draws for both classifiers
                rows.append(dict(seed=seed, sensors=n, nights=30, clf=clf,
                                 **run(crn, truth, hot_cell, undisc, n, 30, se, sp)))
        for nt in NIGHTS:
            rows.append(dict(seed=seed, sensors=N_NIGHTS, nights=nt, clf="FPGA classifier", sweep="nights",
                             **run(rng, truth, hot_cell, undisc, N_NIGHTS, nt, SENS, SPEC)))
    R = pd.DataFrame(rows)
    R["sweep"] = R.get("sweep").fillna("sensors")
    agg = R.groupby(["sweep", "clf", "sensors", "nights"]).mean(numeric_only=True).drop(columns="seed")
    print(agg.round(3).to_string())

    # ---------------- figures ----------------
    def basemap(ax, title):
        for ring in outlines:
            ax.plot(ring[:, 0], ring[:, 1], color="#8b949e", lw=0.6)
        ax.set_xlim(*REGION["lon"]); ax.set_ylim(*REGION["lat"]); ax.set_aspect(1 / COSLAT)
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    def field(vals):
        a = np.full(LAT.shape, np.nan); a[land] = vals
        return a

    ext = [REGION["lon"][0], REGION["lon"][1], REGION["lat"][0], REGION["lat"][1]]
    plt.rcParams.update({"font.size": 10})

    # fig 1: what we know now
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    basemap(ax[0], f"What we know now: {known['records']:,} recordings at {known['sites']:,} sites")
    cov = np.zeros(LAT.shape);
    for (i, j) in cells_any:
        cov[i, j] = 1
    ax[0].imshow(np.where(land, cov, np.nan), extent=ext, origin="lower", cmap="Greys", vmin=-1, vmax=2, alpha=.5)
    ax[0].scatter(m.longitude[~m.res], m.latitude[~m.res], s=4, c="#4dd2ff", label="other bats")
    ax[0].scatter(m.longitude[m.res], m.latitude[m.res], s=6, c="#ff5c5c", label="horseshoe bats (SARSr-CoV reservoirs)")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[0].text(0.01, 0.99, f"{known['cells_with_any_record_pct']}% of 10-km land cells ever recorded\n"
               f"{known['cells_with_reservoir_record_pct']}% have a horseshoe-bat record\n"
               f"{known['records_jun_aug_pct']}% of recordings in Jun-Aug",
               transform=ax[0].transAxes, va="top", fontsize=9, bbox=dict(fc="white", alpha=.8, lw=0))
    ax[1].bar(range(1, 13), months.values, color="#4dd2ff")
    ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND")); ax[1].set_title("Recordings per month (seasonal blind spots)")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "1_known_now.png"), dpi=140); plt.close(fig)

    # fig 2: truth / today / with sensors (seed 0)
    # example = the MEDIAN test world (by undiscovered roosts found with N_MAP sensors), not a cherry-pick
    sub = R[(R.sweep == "sensors") & (R.sensors == N_MAP) & (R.clf == "FPGA classifier")].sort_values("undiscovered_found")
    ex_seed = int(sub.iloc[len(sub) // 2].seed)
    rng = np.random.default_rng(ex_seed)
    truth, hot_xy, hot_cell, undisc = make_world(rng)
    today = run(np.random.default_rng([ex_seed, 0]), truth, hot_cell, undisc, 0, 30, SENS, SPEC, keep=True)
    net = run(np.random.default_rng([ex_seed, N_MAP]), truth, hot_cell, undisc, N_MAP, 30, SENS, SPEC, keep=True)
    print(f"example map: test world {ex_seed} (median); undiscovered found {net['undiscovered_found']:.2f}")
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2))
    hll = np.c_[hot_xy[:, 1] / KM, hot_xy[:, 0] / KM / COSLAT]
    for a, vals, t, note in [
            (ax[0], truth, "Test world (the truth)", "◆ known roosts   ★ undiscovered roosts"),
            (ax[1], sigm(today["mean"]), "Predicted from today's recordings",
             f"undiscovered roosts found: {today['undiscovered_found']*100:.0f}%\n"
             f"area where risk is unknown: {today['unknown_area_pct']:.0f}% (hatched)"),
            (ax[2], sigm(net["mean"]), f"Predicted with {N_MAP} FPGA sensors",
             f"undiscovered roosts found: {net['undiscovered_found']*100:.0f}%\n"
             f"area where risk is unknown: {net['unknown_area_pct']:.0f}% (hatched)")]:
        basemap(a, t)
        a.text(0.02, 0.02, note, transform=a.transAxes, fontsize=9, va="bottom",
               bbox=dict(fc="white", alpha=.85, lw=0))
        im = a.imshow(field(vals), extent=ext, origin="lower", cmap="magma", vmin=0, vmax=1)
    for a, r in [(ax[1], today), (ax[2], net)]:
        unc = field((r["sd"] > 1.0).astype(float))
        a.contourf(LON, LAT, np.nan_to_num(unc), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax[0].scatter(hll[~undisc, 1], hll[~undisc, 0], marker="D", s=22, c="#4dd2ff", edgecolors="k", lw=.5)
    ax[0].scatter(hll[undisc, 1], hll[undisc, 0], marker="*", s=70, c="#3fb950", edgecolors="k", lw=.5)
    ax[1].scatter(m.longitude, m.latitude, s=2, c="#4dd2ff", alpha=.6)
    s_ll = np.c_[net["sensors"][:, 1] / KM, net["sensors"][:, 0] / KM / COSLAT]
    ax[2].scatter(s_ll[:, 1], s_ll[:, 0], marker="^", s=12, c="#3fb950", edgecolors="k", lw=.3)
    cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.01); cb.set_label("share of bat activity from horseshoe bats")
    fig.savefig(os.path.join(OUT, "2_maps.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # fig 3: curves
    S = agg.loc["sensors"]
    fpga, perf = S.loc["FPGA classifier"].reset_index(), S.loc["perfect classifier"].reset_index()
    nt = agg.loc["nights"].loc["FPGA classifier"].reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(fpga.sensors, 100 * fpga.undiscovered_found, "o-", c="#e8590c", label="FPGA classifier (measured errors)")
    ax[0].plot(perf.sensors, 100 * perf.undiscovered_found, "s--", c="#868e96", label="perfect classifier")
    ax[0].plot(fpga.sensors, 100 * fpga.known_found, "o:", c="#1c7ed6", label="known roosts (FPGA)")
    ax[0].plot(fpga.sensors, fpga.high_area_flagged_pct, "^-", c="#c2255c", label="high-risk area flagged (FPGA)")
    ax[0].set_xlabel("number of sensors (30 nights)"); ax[0].set_ylabel("% of roost hotspots found")
    ax[0].set_title("Roost hotspots found"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 100)
    ax[1].plot(fpga.sensors, fpga.unknown_area_pct, "o-", c="#e8590c")
    ax[1].set_xlabel("number of sensors (30 nights)"); ax[1].set_ylabel("% of region")
    ax[1].set_title("Area where risk is still unknown"); ax[1].set_ylim(0, 100)
    ax[2].plot(nt.nights, 100 * nt.undiscovered_found, "o-", c="#e8590c")
    ax[2].set_xscale("log"); ax[2].set_xticks(NIGHTS, NIGHTS)
    ax[2].set_xlabel(f"nights of listening ({N_NIGHTS} sensors)"); ax[2].set_ylabel("% of undiscovered roosts found")
    ax[2].set_title("Where you listen matters more than how long"); ax[2].set_ylim(0, 100)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "3_curves.png"), dpi=140); plt.close(fig)

    out = dict(known_now=known, assumptions=dict(
        passes_per_night=PASSES_PER_NIGHT, sensor_sensitivity=SENS, sensor_specificity=SPEC,
        gp_length_km=ELL_KM, hotspots=N_HOT, seeds=SEEDS, found_rule="P(share>0.4) > 0.7 at roost centre",
        unknown_rule="posterior sd of logit share > 1.0"),
        results=[dict(zip(["sweep", "clf", "sensors", "nights"], k), **{c: round(v, 4) for c, v in r.items()})
                 for k, r in agg.iterrows()])
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
