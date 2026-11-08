#!/usr/bin/env python3
"""
Animate Pteropus (flying fox) tracking data across Myanmar, Thailand and Cambodia.
Usage:
    python make_bat_gif.py
    python make_bat_gif.py --csv bat_monitoring.csv --out bat_monitoring --format gif --fps 12
"""
import argparse
from pathlib import Path as FilePath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Ellipse, FancyBboxPatch

# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default="bat_monitoring.csv")
ap.add_argument("--out", default="bat_monitoring")
ap.add_argument("--format", choices=["gif", "mp4"], default="gif")
ap.add_argument("--step", type=int, default=1, help="Interpolated steps per day")
ap.add_argument("--fps", type=int, default=12, help="Frames per second (lower = slower animation)")
ap.add_argument("--dpi", type=int, default=100)
args = ap.parse_args()

# ----------------------------------------------------------------------------
# Map data
# ----------------------------------------------------------------------------
_shapefile = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')

cambodia_geom = None
for _record in shpreader.Reader(_shapefile).records():
    if _record.attributes.get('NAME') == 'Cambodia' or _record.attributes.get('ADMIN') == 'Cambodia':
        cambodia_geom = _record.geometry
        break

def in_cambodia(xy):
    """Check if points (lon, lat) are inside Cambodia."""
    if cambodia_geom is None:
        return np.zeros(len(xy), dtype=bool)
    return np.array([cambodia_geom.contains(Point(pt[0], pt[1])) for pt in xy])

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
BG = "#ffffff"
LAND = "#f7f7f5"
LAND_TH = "#efeee9"
LAND_KH = "#e9e8f0"
EDGE = "#d5d5d8"
TEXT = "#2b2b2f"
MUTED = "#a4a4ac"
BAT_TH = "#2f6f7a"
BAT_KH = "#e0501c"

XLIM, YLIM = (92.0, 108.5), (9.3, 24.8)

STEPS_PER_DAY = max(1, args.step)
HOLD_FRAMES = 16 
POPUP_HOLD_DAYS = 18.0

# Circle look
BLOB_LAYERS = 24
BLOB_LAYER_ALPHA = 0.065
BLOB_CORE = 0.62       
R_PER_SQRT_N = 0.085    
R_PER_SPREAD = 0.7     
EMA = 0.30        

def smooth(f):
    return f * f * (3 - 2 * f)

def load(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    dates = np.sort(df["date"].unique())
    ids = sorted(df["bat_id"].unique())
    lon = df.pivot(index="date", columns="bat_id", values="lon").loc[dates, ids].to_numpy()
    lat = df.pivot(index="date", columns="bat_id", values="lat").loc[dates, ids].to_numpy()
    pos = np.stack([lon, lat], axis=-1)  # (days, bats, 2)
    origin = df.drop_duplicates("bat_id").set_index("bat_id").loc[ids, "origin_roost"].to_numpy()
    return pd.to_datetime(dates), pos, origin

def position_at(pos, t):
    n = len(pos)
    t = min(max(t, 0.0), n - 1)
    i = int(np.floor(t))
    j = min(i + 1, n - 1)
    f = smooth(t - i)
    return pos[i] * (1 - f) + pos[j] * f

def group_target(xy):
    n = len(xy)
    if n == 0:
        return None, 0.0
    centre = xy.mean(axis=0)
    spread = np.sqrt(((xy - centre) ** 2).sum(axis=1).mean())
    return centre, R_PER_SQRT_N * np.sqrt(n) + R_PER_SPREAD * spread

# ----------------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------------
dates, pos, origin = load(args.csv)
n_days, n_bats, _ = pos.shape
roosts = list(dict.fromkeys(origin))     
group_names = roosts + ["Cambodia"]        
n_groups = len(group_names)

t_main = np.arange(0, (n_days - 1) * STEPS_PER_DAY + 1) / STEPS_PER_DAY
timeline = np.concatenate([t_main, np.full(HOLD_FRAMES, t_main[-1])])

centres = np.zeros((len(timeline), n_groups, 2))
radii = np.zeros((len(timeline), n_groups))
cur_c = np.full((n_groups, 2), np.nan)
cur_r = np.zeros(n_groups)
t_alert = None

for k, t in enumerate(timeline):
    xy = position_at(pos, t)
    kh = in_cambodia(xy)
    if t_alert is None and kh.any():
        t_alert = t
    for g, name in enumerate(group_names):
        members = xy[kh] if name == "Cambodia" else xy[(origin == name) & ~kh]
        c, r = group_target(members)
        if c is not None:
            cur_c[g] = c if np.isnan(cur_c[g]).any() or cur_r[g] < 0.02 else cur_c[g] + EMA * (c - cur_c[g])
        cur_r[g] += EMA * (r - cur_r[g])
        if cur_r[g] < 0.01:
            cur_r[g] = 0.0
        centres[k, g] = np.nan_to_num(cur_c[g])
        radii[k, g] = cur_r[g]

# ----------------------------------------------------------------------------
# Figure setup
# ----------------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(8.4, 9.0), dpi=100, facecolor=BG)
ax = fig.add_axes([0.02, 0.02, 0.96, 0.88], projection=ccrs.PlateCarree(), facecolor=BG)
ax.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]], crs=ccrs.PlateCarree())
ax.axis("off")

# Render country fills and borders using Cartopy shapereader (50m resolution)
special = {"Thailand": LAND_TH, "Cambodia": LAND_KH}

reader = shpreader.Reader(_shapefile)
for record in reader.records():
    name = record.attributes.get('NAME', '')
    geom = record.geometry
    fc = special.get(name, LAND)
    ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=fc, edgecolor='none', zorder=1)

reader = shpreader.Reader(_shapefile)
for record in reader.records():
    geom = record.geometry
    ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor=EDGE, linewidth=0.6, zorder=1.5)

lab = dict(color=MUTED, fontsize=7, ha="center", va="center", transform=ccrs.PlateCarree(), zorder=2)
for text, x, y in [("MYANMAR", 95.0, 24.1), ("THAILAND", 101.3, 17.9), ("LAOS", 103.9, 19.6),
                   ("VIETNAM", 106.7, 21.4), ("CAMBODIA", 105.5, 12.1), ("CHINA", 103.0, 24.3)]:
    ax.text(x, y, text, **lab)

blobs = []
for name in group_names:
    colour = BAT_KH if name == "Cambodia" else BAT_TH
    layers = []
    for _ in range(BLOB_LAYERS):
        e = Ellipse((0, 0), 0, 0, transform=ccrs.PlateCarree(), fc=colour, ec="none", alpha=BLOB_LAYER_ALPHA,
                    zorder=4, visible=False)
        ax.add_patch(e)
        layers.append(e)
    blobs.append(layers)

# Header and key
fig.text(0.04, 0.955, "Pteropus monitoring", fontsize=12, color=TEXT, weight="bold", va="center")
fig.text(0.04, 0.925, "Myanmar, Thailand & Cambodia - synthetic demo data", fontsize=8.5, color=MUTED, va="center")
date_txt = fig.text(0.96, 0.955, "", fontsize=11, color=TEXT, ha="right", va="center")


def popup_alpha(t):
    if t_alert is None:
        return 0.0
    dt = t - t_alert
    if dt < 0:
        return 0.0
    if dt < 0.5:
        return smooth(dt / 0.5)
    if dt < POPUP_HOLD_DAYS:
        return 1.0
    if dt < POPUP_HOLD_DAYS + 1.0:
        return 1.0 - smooth(dt - POPUP_HOLD_DAYS)
    return 0.0

# ----------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------
def update(k):
    t = timeline[k]
    for g in range(n_groups):
        r = radii[k, g]
        cx_, cy_ = centres[k, g]
        lat_corr = np.cos(np.radians(cy_)) if not np.isnan(cy_) else np.cos(np.radians(17))
        lat_corr = max(lat_corr, 0.1)
        for i, e in enumerate(blobs[g]):
            if r <= 0:
                e.set_visible(False)
                continue
            rr = r * (BLOB_CORE + (1 - BLOB_CORE) * (i + 1) / BLOB_LAYERS)
            e.set_visible(True)
            e.set_center((cx_, cy_))
            e.set_width(2 * rr)
            e.set_height(2 * rr / lat_corr)
    day = dates[int(np.floor(min(t, n_days - 1)))]
    date_txt.set_text(day.strftime("%d %b %Y"))
    return []

anim = FuncAnimation(fig, update, frames=len(timeline), blit=False)

out_file = f"{args.out}.{args.format}"
if args.format == "gif":
    writer = PillowWriter(fps=args.fps)
else:
    writer = FFMpegWriter(fps=args.fps, bitrate=3000)

print(f"Rendering {len(timeline)} frames at {args.fps} fps -> {out_file} ...")
anim.save(out_file, writer=writer, dpi=args.dpi, savefig_kwargs={"facecolor": BG})
plt.close(fig)

if t_alert is not None:
    d = dates[int(np.floor(t_alert))].strftime("%Y-%m-%d")
    print(f"Warning triggered around {d}: first bat crossed into Cambodia.")
print(f"Saved {out_file} ({len(timeline)} frames)")