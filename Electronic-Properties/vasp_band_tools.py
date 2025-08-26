#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vasp_band_tools.py
------------------
Plot VASP band structures (and projections) from `vasprun.xml` (+ `KPOINTS`).

Features
- Band structure with E_F = 0 alignment
- Optional element- or orbital-projected band character (thickness coloring)
- Spin-polarized support
- CSV export of band data (k-distance, segment, band index, energy, spin)
- High-resolution PNG (and optional SVG) outputs

Author: ChatGPT Assistant
License: MIT
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin, OrbitalType
from pymatgen.electronic_structure.plotter import BSPlotter, BSPlotterProjected


def load_band_structure(vasprun_path: Path,
                        kpoints_path: Optional[Path],
                        parse_projected: bool = True):
    """
    Load a BandStructure object from vasprun.xml (+KPOINTS for line-mode).

    Tries BSVasprun first (better for band parsing), falls back to Vasprun.get_band_structure().
    """
    # First try BSVasprun
    try:
        bsv = BSVasprun(str(vasprun_path), parse_projected_eigen=parse_projected, parse_potcar_file=False)
        bs = bsv.get_band_structure(kpoints_filename=str(kpoints_path) if kpoints_path else "KPOINTS",
                                    line_mode=True,
                                    efermi=bsv.efermi)
        return bs, bsv.efermi
    except Exception as e:
        # Fallback to Vasprun
        vr = Vasprun(str(vasprun_path),
                     parse_dos=False,
                     parse_eigen=True,
                     parse_projected_eigen=parse_projected,
                     parse_potcar_file=False)
        bs = vr.get_band_structure(kpoints_filename=str(kpoints_path) if kpoints_path else "KPOINTS",
                                   efermi=vr.efermi,
                                   line_mode=True)
        return bs, vr.efermi


def bs_to_dataframe(bs) -> pd.DataFrame:
    """
    Flatten band structure into a tidy DataFrame.
    Columns: segment, k_index, k_distance, band_index, spin, energy_eV
    """
    plotter = BSPlotter(bs)
    data = plotter.bs_plot_data()

    frames = []
    # For each spin channel
    spins = [Spin.up]
    if bs.is_spin_polarized:
        spins = [Spin.up, Spin.down]

    # data["distances"] is list of segments; each is list of k_dists
    # data["energy"][spin] is list of bands; each band is list of segments; each segment list matches distances
    for spin in spins:
        spin_key = "up" if spin == Spin.up else "down"
        distances = data["distances"]
        energies_all_bands = data["energy"][spin]
        for band_idx, band_segments in enumerate(energies_all_bands):
            for seg_idx, (seg_kdists, seg_energies) in enumerate(zip(distances, band_segments)):
                # seg_kdists: list of floats
                # seg_energies: list of floats
                n = len(seg_kdists)
                df = pd.DataFrame({
                    "segment": seg_idx,
                    "k_index": np.arange(n),
                    "k_distance": seg_kdists,
                    "band_index": band_idx,
                    "spin": spin_key,
                    "energy_eV": seg_energies
                })
                frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def save_plain_plot(bs, out_png: Path, emin: Optional[float], emax: Optional[float], dpi: int = 600, also_svg: bool = False):
    plotter = BSPlotter(bs)
    fig = plotter.get_plot(ylim=(emin, emax))
    fig.gca().axhline(0.0, linestyle="--", linewidth=1)  # EF=0
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    if also_svg:
        fig.savefig(out_png.with_suffix(".svg"))
    plt.close(fig)


def save_projected_plot(bs,
                        out_png: Path,
                        project_on: str = "elements",
                        elements: Optional[List[str]] = None,
                        orbitals: Optional[List[str]] = None,
                        emin: Optional[float] = None,
                        emax: Optional[float] = None,
                        dpi: int = 600,
                        also_svg: bool = False):
    """
    Save projected band plot using BSPlotterProjected.
    project_on: "elements" or "orbitals"
    elements: list of element symbols to include (default: all)
    orbitals: list of orbitals e.g. ["s", "p", "d"]
    """
    p = BSPlotterProjected(bs)
    fig = None

    # Determine projections
    if project_on == "elements":
        if elements:
            fig = p.get_elt_projected_plots(elements=elements, ylim=(emin, emax))
        else:
            fig = p.get_elt_projected_plots(ylim=(emin, emax))
    elif project_on == "orbitals":
        # Map strings to OrbitalType
        if orbitals is None:
            orbitals = ["s", "p", "d"]
        orbital_types = []
        for o in orbitals:
            o = o.lower()
            if o == "s":
                orbital_types.append(OrbitalType.s)
            elif o == "p":
                orbital_types.append(OrbitalType.p)
            elif o == "d":
                orbital_types.append(OrbitalType.d)
            elif o == "f":
                orbital_types.append(OrbitalType.f)
        fig = p.get_orb_projected_plots(orbitals=orbital_types, ylim=(emin, emax))
    else:
        raise ValueError("project_on must be 'elements' or 'orbitals'")

    ax = fig.gca()
    ax.axhline(0.0, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    if also_svg:
        fig.savefig(out_png.with_suffix(".svg"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot band structure from VASP vasprun.xml (+KPOINTS).")
    ap.add_argument("vasprun", type=str, help="Path to vasprun.xml")
    ap.add_argument("--kpoints", type=str, default="KPOINTS", help="KPOINTS file for the band path (line-mode)")
    ap.add_argument("--outdir", type=str, default="band_output", help="Output directory")
    ap.add_argument("--emin", type=float, default=None, help="Min energy (eV) relative to E_F for y-axis")
    ap.add_argument("--emax", type=float, default=None, help="Max energy (eV) relative to E_F for y-axis")
    ap.add_argument("--dpi", type=int, default=600, help="Figure DPI (default 600)")
    ap.add_argument("--svg", action="store_true", help="Also save SVG")
    ap.add_argument("--no-proj", action="store_true", help="Disable projected plot even if projections exist")

    # Projection options
    ap.add_argument("--project", choices=["elements", "orbitals"], default=None,
                    help="Type of projection for band character (requires projections in vasprun.xml/PROCAR)")
    ap.add_argument("--elements", nargs="*", default=None, help="Element symbols to include in projection (e.g., C O Fe)")
    ap.add_argument("--orbitals", nargs="*", default=None, help="Orbitals to include (e.g., s p d f)")

    args = ap.parse_args()

    vasprun_path = Path(args.vasprun).expanduser().resolve()
    kpoints_path = Path(args.kpoints).expanduser().resolve() if args.kpoints else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not vasprun_path.exists():
        raise FileNotFoundError(f"vasprun.xml not found at: {vasprun_path}")
    if kpoints_path and not kpoints_path.exists():
        print(f"[WARN] KPOINTS not found at {kpoints_path}. Trying default 'KPOINTS' in CWD if available.")

    # Load BS
    bs, efermi = load_band_structure(vasprun_path, kpoints_path, parse_projected=(args.project is not None))

    # CSV export
    band_df = bs_to_dataframe(bs)
    band_df.to_csv(outdir / "bands.csv", index=False)

    # Plain band plot
    save_plain_plot(bs, outdir / "band_plain.png", emin=args.emin, emax=args.emax, dpi=args.dpi, also_svg=args.svg)

    # Projected band plot (optional)
    if not args.no_proj and args.project is not None:
        try:
            save_projected_plot(
                bs,
                out_png=outdir / f"band_projected_{args.project}.png",
                project_on=args.project,
                elements=args.elements,
                orbitals=args.orbitals,
                emin=args.emin,
                emax=args.emax,
                dpi=args.dpi,
                also_svg=args.svg
            )
        except Exception as e:
            print(f"[WARN] Projected plot failed ({e}). Falling back to plain band plot only.")

    # Band gap summary
    gap = bs.get_band_gap()
    direct = gap.get("direct", False)
    gap_e = gap.get("energy", None)
    print("Band gap summary:", gap)
    if gap_e is not None:
        print(f"Gap: {gap_e:.3f} eV ({'direct' if direct else 'indirect'})")
    print(f"Fermi level from vasprun: {efermi:.3f} eV (energies plotted relative to this)")
    print(f"Outputs saved to: {outdir}")

if __name__ == "__main__":
    main()
