#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vasp_dos_tools.py
-----------------
Parse and analyze Density of States (DOS/PDOS) from VASP `vasprun.xml`.

Features
- Total DOS with E_F = 0 alignment
- Element-resolved s/p/d DOS (summed over sites of each element)
- Optional site-resolved s/p/d DOS for user-specified site indices
- Gaussian smoothing (optional)
- CSV export of all plotted curves
- High-resolution matplotlib figures

Author: ChatGPT Assistant
License: MIT
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.core import Spin, OrbitalType

# -------------------------
# Helper utilities
# -------------------------

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def smear_dos(dos_obj, sigma: Optional[float] = None):
    """Return densities (dict[Spin, np.ndarray]) with optional Gaussian smoothing."""
    if sigma and sigma > 0:
        return dos_obj.get_smeared_densities(sigma)
    return dos_obj.densities

def dos_to_dataframe(energies: np.ndarray, densities: Dict[Spin, np.ndarray]) -> pd.DataFrame:
    """Build a tidy DataFrame with columns: Energy(eV), DOS_up, DOS_down (if present)."""
    data = {"Energy (eV)": energies}
    if Spin.up in densities:
        data["DOS_up"] = densities[Spin.up]
    if Spin.down in densities:
        data["DOS_down"] = densities[Spin.down]
    return pd.DataFrame(data)

def sum_spin(densities: Dict[Spin, np.ndarray]) -> np.ndarray:
    """Return spin-summed DOS."""
    out = np.zeros_like(next(iter(densities.values())))
    for arr in densities.values():
        out += arr
    return out

def select_energy_window(energies: np.ndarray, emin: Optional[float], emax: Optional[float]) -> np.ndarray:
    """Return a boolean mask selecting energies within [emin, emax] if provided."""
    if emin is None and emax is None:
        return np.ones_like(energies, dtype=bool)
    emin_val = -np.inf if emin is None else float(emin)
    emax_val = np.inf if emax is None else float(emax)
    return (energies >= emin_val) & (energies <= emax_val)

# -------------------------
# Plotting
# -------------------------

def plot_total_and_element_spd(vasprun_path: Path,
                               outdir: Path,
                               elements: Optional[Iterable[str]] = None,
                               sigma: Optional[float] = None,
                               emin: Optional[float] = None,
                               emax: Optional[float] = None,
                               dpi: int = 600,
                               spin_sum: bool = True) -> None:
    """Plot total DOS and element-resolved s/p/d DOS."""
    vr = Vasprun(str(vasprun_path), parse_potcar_file=False)
    cdos = vr.complete_dos
    efermi = vr.efermi

    # Energies relative to EF
    energies = cdos.energies - efermi
    mask = select_energy_window(energies, emin, emax)
    energies = energies[mask]

    # Total DOS
    total_dens = smear_dos(cdos, sigma)
    total_df = dos_to_dataframe(energies, {s: d[mask] for s, d in total_dens.items()})
    total_df.to_csv(outdir / "total_dos.csv", index=False)

    plt.figure(figsize=(5, 4), dpi=dpi)
    if spin_sum:
        plt.plot(energies, sum_spin({s: d[mask] for s, d in total_dens.items()}), label="Total DOS")
    else:
        for s, d in total_dens.items():
            label = "Total DOS (up)" if s == Spin.up else "Total DOS (down)"
            plt.plot(energies, d[mask], label=label)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Energy – $E_F$ (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.title("Total DOS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "total_dos.png")
    plt.close()

    # Element-resolved SPD
    elem_spd = cdos.get_element_spd_dos()  # {Element: {OrbitalType.s/p/d: Dos}}
    # Build a symbol->spd_map dict for robust lookup
    elem_spd_by_symbol = {str(el): spd for el, spd in elem_spd.items()}

    if elements is None or len(list(elements)) == 0:
        # default: all elements present
        elements = list(elem_spd_by_symbol.keys())

    for el in elements:
        spd_map = elem_spd_by_symbol.get(el)
        if spd_map is None:
            print(f"[WARN] Element '{el}' not found in structure; skipping.")
            continue

        plt.figure(figsize=(5, 4), dpi=dpi)
        csv_cols = {"Energy (eV)": energies}
        for orb in [OrbitalType.s, OrbitalType.p, OrbitalType.d]:
            if orb in spd_map:
                dens = smear_dos(spd_map[orb], sigma)
                if spin_sum:
                    y = sum_spin(dens)[mask]
                    plt.plot(energies, y, label=f"{el} {orb.name}")
                    csv_cols[f"{el}_{orb.name}"] = y
                else:
                    for s, d in dens.items():
                        label = f"{el} {orb.name} ({'up' if s == Spin.up else 'down'})"
                        plt.plot(energies, d[mask], label=label)
                        csv_cols[label] = d[mask]
        if len(csv_cols) > 1:
            pd.DataFrame(csv_cols).to_csv(outdir / f"element_spd_{el}.csv", index=False)

        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Energy – $E_F$ (eV)")
        plt.ylabel("DOS (states/eV)")
        plt.title(f"{el}: s/p/d-projected DOS")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"element_spd_{el}.png")
        plt.close()


def plot_site_spd(vasprun_path: Path,
                  outdir: Path,
                  sites: Iterable[int],
                  sigma: Optional[float] = None,
                  emin: Optional[float] = None,
                  emax: Optional[float] = None,
                  dpi: int = 600,
                  spin_sum: bool = True) -> None:
    """Plot s/p/d DOS for specific site indices (0-based)."""
    vr = Vasprun(str(vasprun_path), parse_potcar_file=False)
    cdos = vr.complete_dos
    efermi = vr.efermi

    energies = cdos.energies - efermi
    mask = select_energy_window(energies, emin, emax)
    energies = energies[mask]

    for idx in sites:
        try:
            site = cdos.structure[idx]
        except IndexError:
            print(f"[WARN] Site index {idx} out of range; skipping.")
            continue

        spd_map = cdos.get_site_spd_dos(site)  # {OrbitalType: Dos}
        plt.figure(figsize=(5, 4), dpi=dpi)
        csv_cols = {"Energy (eV)": energies}
        for orb in [OrbitalType.s, OrbitalType.p, OrbitalType.d]:
            if orb in spd_map:
                dens = smear_dos(spd_map[orb], sigma)
                if spin_sum:
                    y = sum_spin(dens)[mask]
                    plt.plot(energies, y, label=f"site {idx} {orb.name}")
                    csv_cols[f"site{idx}_{orb.name}"] = y
                else:
                    for s, d in dens.items():
                        label = f"site {idx} {orb.name} ({'up' if s == Spin.up else 'down'})"
                        plt.plot(energies, d[mask], label=label)
                        csv_cols[label] = d[mask]

        if len(csv_cols) > 1:
            pd.DataFrame(csv_cols).to_csv(outdir / f"site_spd_{idx}.csv", index=False)

        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Energy – $E_F$ (eV)")
        plt.ylabel("DOS (states/eV)")
        plt.title(f"Site {idx}: s/p/d-projected DOS")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"site_spd_{idx}.png")
        plt.close()

# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Parse and plot DOS/PDOS from VASP vasprun.xml")
    p.add_argument("vasprun", type=str, help="Path to vasprun.xml")
    p.add_argument("--outdir", type=str, default="dos_output", help="Output directory")
    p.add_argument("--elements", nargs="*", default=None, help="Element symbols to plot (default: all present)")
    p.add_argument("--sites", nargs="*", type=int, default=None, help="0-based site indices to plot (e.g., 0 5 12)")
    p.add_argument("--sigma", type=float, default=None, help="Gaussian smearing width in eV (e.g., 0.10)")
    p.add_argument("--emin", type=float, default=None, help="Min energy (eV) relative to E_F (e.g., -6)")
    p.add_argument("--emax", type=float, default=None, help="Max energy (eV) relative to E_F (e.g., 6)")
    p.add_argument("--dpi", type=int, default=600, help="Figure DPI (default: 600)")
    p.add_argument("--no-spinsum", action="store_true", help="If set, plot spin-up and spin-down separately")

    args = p.parse_args()

    vasprun_path = Path(args.vasprun).expanduser().resolve()
    outdir = Path(args.outdir)

    if not vasprun_path.exists():
        raise FileNotFoundError(f"vasprun.xml not found at: {vasprun_path}")

    ensure_outdir(outdir)

    spin_sum = not args.no_spinsum

    # Total & element-resolved
    plot_total_and_element_spd(
        vasprun_path=vasprun_path,
        outdir=outdir,
        elements=args.elements,
        sigma=args.sigma,
        emin=args.emin,
        emax=args.emax,
        dpi=args.dpi,
        spin_sum=spin_sum
    )

    # Site-resolved (optional)
    if args.sites:
        plot_site_spd(
            vasprun_path=vasprun_path,
            outdir=outdir,
            sites=args.sites,
            sigma=args.sigma,
            emin=args.emin,
            emax=args.emax,
            dpi=args.dpi,
            spin_sum=spin_sum
        )

    print(f"Done. Results saved under: {outdir}")


if __name__ == "__main__":
    main()
