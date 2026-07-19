import torch
import numpy as np
import pandas as pd

from typing import Union
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from matplotlib.colors import LogNorm
from scipy import ndimage
from typing import Optional, Union, Sequence
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from tools.plotting import plot_radial_PSF_profiles, _cube_to_RGB_array

"""
This module is used to manage the multi-source simulations. It contains functions to 
- detect sources in an image,
- extract multiple source images from a single image/cube with multiple sources,
- merge multiple images into a single image,
- visualize sources.
"""


@dataclass
class SourcesSubset:
    ids: list[int]
    table: pd.DataFrame
    imgs_sparse:   list | None
    slices_local:  list
    slices_global: list
    spectra_sparse: torch.Tensor | None
    spectra_full:   torch.Tensor | None
    spectra_sparse_true: Optional[torch.Tensor] = None
    spectra_full_true :  Optional[torch.Tensor] = None
    spectra_res_sparse:  Optional[torch.Tensor] = None
    spectra_res_full:    Optional[torch.Tensor] = None
    proximity_table:     Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.ids)
    
    def clone(self, light_weight=False) -> "SourcesSubset":
        """ Light-weight copy creates a copy of the subset without the potentially large image and spectra data when not needed. """
        return SourcesSubset(
            ids   = self.ids.copy(),
            table = self.table.copy(),
            imgs_sparse         = self.imgs_sparse.copy() if not light_weight else None,
            slices_local        = self.slices_local.copy(),
            slices_global       = self.slices_global.copy(),
            spectra_sparse      = self.spectra_sparse.clone() if not light_weight else None,
            spectra_sparse_true = self.spectra_sparse_true.clone() if self.spectra_sparse_true is not None and not light_weight else None,
            spectra_full        = self.spectra_full.clone() if not light_weight else None,
            spectra_full_true   = self.spectra_full_true.clone() if self.spectra_full_true is not None and not light_weight else None,
            spectra_res_sparse  = self.spectra_res_sparse.clone() if self.spectra_res_sparse is not None and not light_weight else None,
            spectra_res_full    = self.spectra_res_full.clone() if self.spectra_res_full is not None and not light_weight else None,
            proximity_table     = self.proximity_table.copy() if self.proximity_table is not None else None
        )

    def rebase_coords(self, crop_ROI, filter_empty=True) -> "SourcesSubset":
        """
        Return a new SourcesSubset whose slices_local / slices_global are re-expressed
        relative to a crop of the original image.

        Parameters
        ----------
        crop_ROI : (slice_y, slice_x)
            The crop as a pair of slices, e.g. ``(slice(64, 192), slice(32, 160))``.
            ``slice.stop`` may be ``None`` to indicate "until the end" (no upper clipping).
        filter_empty : bool
            If True (default), sources with no overlap with the crop are dropped and all
            other fields (table, imgs_sparse, spectra_*, proximity_table) are subset
            accordingly.  If False, all entries are kept with zero-size slices for
            non-overlapping sources (harmless no-ops in ``add_ROIs``).

        Returns
        -------
        SourcesSubset  - new subset with rebased coordinate lists (and optionally filtered).
        """
        slice_y, slice_x = crop_ROI
        y_start = slice_y.start if slice_y.start is not None else 0
        x_start = slice_x.start if slice_x.start is not None else 0
        y_end   = slice_y.stop
        x_end   = slice_x.stop

        new_local  = []
        new_global = []
        kept_ids   = []

        for idx, (local, glob) in enumerate(zip(self.slices_local, self.slices_global)):
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local
            (y_min_img, y_max_img), (x_min_img, x_max_img) = glob

            # Shift global coords to be relative to crop origin
            ny_min = y_min_img - y_start
            ny_max = y_max_img - y_start
            nx_min = x_min_img - x_start
            nx_max = x_max_img - x_start

            # Advance local coords to skip the part of the ROI that falls before the crop
            new_y_min_roi = y_min_roi + max(0, -ny_min)
            new_x_min_roi = x_min_roi + max(0, -nx_min)

            # Clip global lower bound to 0
            ny_min = max(0, ny_min)
            nx_min = max(0, nx_min)

            # Trim local / global upper bounds at the crop edge
            if y_end is not None:
                new_y_max_roi = y_max_roi - max(0, ny_max - (y_end - y_start))
                ny_max = min(ny_max, y_end - y_start)
            else:
                new_y_max_roi = y_max_roi

            if x_end is not None:
                new_x_max_roi = x_max_roi - max(0, nx_max - (x_end - x_start))
                nx_max = min(nx_max, x_end - x_start)
            else:
                new_x_max_roi = x_max_roi

            if filter_empty and (ny_min >= ny_max or nx_min >= nx_max or
                                  new_y_min_roi >= new_y_max_roi or new_x_min_roi >= new_x_max_roi):
                continue

            new_local.append(((new_y_min_roi, new_y_max_roi), (new_x_min_roi, new_x_max_roi)))
            new_global.append(((ny_min, ny_max), (nx_min, nx_max)))
            kept_ids.append(idx)

        # Build the filtered subset if filter_empty=True, otherwise keep all entries with adjusted slices
        k = kept_ids
        return SourcesSubset(
            ids                 = [self.ids[i] for i in k],
            table               = self.table.iloc[k].reset_index(drop=True) if self.table is not None else None,
            imgs_sparse         = [self.imgs_sparse[i] for i in k] if self.imgs_sparse is not None else None,
            slices_local        = new_local,
            slices_global       = new_global,
            spectra_sparse      = self.spectra_sparse[k] if self.spectra_sparse is not None else None,
            spectra_sparse_true = self.spectra_sparse_true[k] if self.spectra_sparse_true is not None else None,
            spectra_full        = self.spectra_full[k]   if self.spectra_full   is not None else None,
            spectra_full_true   = self.spectra_full_true[k] if self.spectra_full_true   is not None else None,
            spectra_res_sparse  = self.spectra_res_sparse[k] if self.spectra_res_sparse is not None else None,
            spectra_res_full    = self.spectra_res_full[k]   if self.spectra_res_full   is not None else None,
            proximity_table     = self.proximity_table[np.ix_(k, k)] if self.proximity_table is not None else None,
        )


@dataclass
class SourcesData:
    """All source-indexed arrays. Row order is the canonical source ID order."""
    table: pd.DataFrame
    imgs_sparse:   list
    slices_local:  list
    slices_global: list
    spectra_full:   torch.Tensor
    spectra_sparse: torch.Tensor
    spectra_sparse_true: Optional[torch.Tensor] = None
    spectra_full_true :  Optional[torch.Tensor] = None
    spectra_res_full:    Optional[torch.Tensor] = None
    spectra_res_sparse:  Optional[torch.Tensor] = None
    proximity_table:     Optional[np.ndarray]   = None

    def __post_init__(self) -> None:
        self.table = self.table.reset_index(drop=True).copy()
        self.table.index.name = "src_id"
        self.table["src_id"] = np.arange(len(self.table), dtype=int)
        self.create_proximity_table() # Precompute the proximity table for efficient spatial queries later  

    def __len__(self) -> int:
        return len(self.table)

    def index(self, src_ids: Optional[Union[int, np.integer, Sequence[int]]] = None) -> list[int]:
        """Normalize src_ids to an explicit validated list. None returns all source IDs."""
        if src_ids is None:
            return list(range(len(self)))
        if isinstance(src_ids, (int, np.integer)):
            src_ids = [int(src_ids)]
        ids = [int(i) for i in src_ids]
        bad = [i for i in ids if i < 0 or i >= len(self)]
        if bad:
            raise IndexError(f"Source IDs out of range: {bad}; valid range is [0, {len(self) - 1}]")
        return ids

    def select(self, src_ids: Optional[Union[int, Sequence[int]]]) -> "SourcesSubset":
        ids = self.index(src_ids)
        return SourcesSubset(
            ids   = ids,
            table = self.table.iloc[ids],
            imgs_sparse         = [self.imgs_sparse[i]   for i in ids],
            slices_local        = [self.slices_local[i]  for i in ids],
            slices_global       = [self.slices_global[i] for i in ids],
            spectra_sparse      = self.spectra_sparse[ids],
            spectra_sparse_true = self.spectra_sparse_true[ids] if self.spectra_sparse_true is not None else None,
            spectra_full        = self.spectra_full[ids],
            spectra_full_true   = self.spectra_full_true[ids] if self.spectra_full_true is not None else None,
            spectra_res_sparse  = self.spectra_res_sparse[ids] if self.spectra_res_sparse is not None else None,
            spectra_res_full    = self.spectra_res_full[ids]   if self.spectra_res_full is not None   else None,
            proximity_table     = self.proximity_table[np.ix_(ids, ids)] if self.proximity_table is not None else None
        )

    def create_proximity_table(self) -> None:
        """Vectorized squared-distance matrix between all source pairs. Returns [N, N] array in pix^2."""
        coords = self.table[['x_peak', 'y_peak']].to_numpy()
        diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 2]
        self.proximity_table = np.einsum('ijk,ijk->ij', diff, diff) # [N, N]

    def get_N_closest_sources(self, i_src: int, N: int) -> "SourcesSubset":
        """Return indices of the n closest sources to i_src, sorted by ascending distance."""
        distances = self.proximity_table[i_src]
        indices = np.argpartition(distances, min(N, len(distances) - 1))[:N]
        return self.select(indices[np.argsort(distances[indices])].tolist())

    def sources_within_radius(self, i_src: int, radius: float, d_offset: float = 0.0) -> "SourcesSubset":
        """Return indices of sources within radius + d_offset of i_src, sorted by ascending distance."""
        distances = self.proximity_table[i_src]
        indices = np.where(distances <= (radius + d_offset) ** 2)[0]
        return self.select(indices[np.argsort(distances[indices])].tolist())

    def select_sources_in_tile(
        self,
        ROI: tuple,   # (slice_y, slice_x) - same format as slices_global entries
        d_offset: float,
        N: int | None = None
    ) -> SourcesSubset:
        """
        Return source indices inside the tile (plus sources within d_offset of its boundary).

        ROI is a (slice_y, slice_x) pair matching the slices_global format, e.g.
            (slice(y_min, y_max), slice(x_min, x_max))
        If N is None, all sources within ROI + d_offset are returned sorted by brightness.
        If N is given and fewer sources qualify, remaining slots are filled by proximity to
        the nearest in-tile source. When the tile has more than N sources, the brightest N are kept.
        """
        sources_pos = self.table[['x_peak', 'y_peak']].to_numpy()
        slice_y, slice_x = ROI
        y_min, y_max = slice_y.start, slice_y.stop
        x_min, x_max = slice_x.start, slice_x.stop

        in_tile_mask = (
            (sources_pos[:, 0] >= x_min) & (sources_pos[:, 0] <= x_max) &
            (sources_pos[:, 1] >= y_min) & (sources_pos[:, 1] <= y_max)
        )
        in_tile_idx = np.where(in_tile_mask)[0].tolist()
        outside_idx = np.where(~in_tile_mask)[0]

        outside_pos = sources_pos[outside_idx]
        dx = np.maximum(0, np.maximum(x_min - outside_pos[:, 0], outside_pos[:, 0] - x_max))
        dy = np.maximum(0, np.maximum(y_min - outside_pos[:, 1], outside_pos[:, 1] - y_max))
        near_mask = np.hypot(dx, dy) <= d_offset

        def _by_brightness(idx_list):
            vals = self.table.iloc[idx_list]['peak_value'].to_numpy()
            return [idx_list[i] for i in np.argsort(vals)[::-1]]

        # When N is not given, return all sources within ROI + d_offset, sorted by brightness
        if N is None:
            in_idx   = _by_brightness(in_tile_idx)
            near_idx = _by_brightness(outside_idx[near_mask].tolist())
            return self.select(in_idx + [i for i in near_idx if i not in set(in_idx)])

        # More sources than needed -> keep brightest
        if len(in_tile_idx) >= N:
            return self.select(_by_brightness(in_tile_idx)[:N])

        # Fewer sources than needed -> pad with nearby sources
        near_idx = _by_brightness(outside_idx[near_mask].tolist())

        all_idx = in_tile_idx + near_idx
        n_needed = N - len(all_idx)

        if n_needed > 0:
            remaining = np.setdiff1d(np.arange(len(self.table)), all_idx)
            if len(in_tile_idx) > 0:
                dists = self.proximity_table[in_tile_idx[0], remaining]
            else:
                dists = -self.table.iloc[remaining]['peak_value'].to_numpy()
            all_idx += remaining[np.argsort(dists)][:n_needed].tolist()

        return self.select(all_idx[:N])

    def subset(self, src_ids: Optional[Union[int, Sequence[int]]] = None) -> "SourcesSubset":
        """Return a subset view for the requested source IDs."""
        return self.select(src_ids)

    def __getitem__(self, src_ids: Union[int, np.integer, Sequence[int], slice]) -> "SourcesSubset":
        """Enable bracket-based source subset selection."""
        if isinstance(src_ids, slice):
            return self.select(list(range(len(self)))[src_ids])
        return self.select(src_ids)

    def delete(self, src_ids: Union[int, np.integer, Sequence[int]]) -> None:
        """Remove source(s) by ID in-place and rebuild the proximity table."""
        if isinstance(src_ids, (int, np.integer)):
            src_ids = [int(src_ids)]
        ids_to_delete = set(int(i) for i in src_ids)
        bad = [i for i in ids_to_delete if i < 0 or i >= len(self)]
        if bad:
            raise IndexError(f"Source IDs out of range: {bad}; valid range is [0, {len(self) - 1}]")

        keep = [i for i in range(len(self)) if i not in ids_to_delete]

        self.table = self.table.iloc[keep].reset_index(drop=True)
        self.table.index.name = "src_id"
        self.table["src_id"] = np.arange(len(self.table), dtype=int)

        self.imgs_sparse   = [self.imgs_sparse[i]   for i in keep] if self.imgs_sparse   is not None else None
        self.slices_local  = [self.slices_local[i]  for i in keep]
        self.slices_global = [self.slices_global[i] for i in keep]

        for attr in ('spectra_sparse', 'spectra_full', 'spectra_sparse_true',
                     'spectra_full_true', 'spectra_res_sparse', 'spectra_res_full'):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val[keep])

        self.create_proximity_table()



def DisplayField(
    cube,
    *,
    sources = None,
    wavelengths = None,
    cmap = 'gray',
    show_markers = True,
    show_ids = False,
    show_weights = False,
    roi = None,
    title = None,
    norm  = None,
    vmin  = None,
    vmax  = None,
    scale = 'log',
    saturation = 1.0,
    contrast   = 1.0,
    wb_shift   = 0.0,
    mg_shift   = 0.0,
    marker_shape = 'circle',
    marker_color = 'gold',
    marker_size = None,
    min_marker_size = 4,
    max_marker_size = 20,
    show = True,
    ax = None,
    figsize = (6, 6),
):
    """
    Unified display: render a spectral cube with a colormap (∑ over λ) or as an RGB image (wavelength→colour mapping), with optional source overlays.

    Parameters
    ----------
    cube : array-like or torch.Tensor, shape (N_λ, H, W) or (H, W)
    sources : SourcesData | SourcesSubset | pd.DataFrame | None
    wavelengths : array-like or None
        Physical wavelengths [nm] for RGB mapping. None → uniform 440-750 nm.
    cmap : str or None
        ``None``  → RGB colour mapping via wavelength-to-colour weights.
        Any Matplotlib colormap name (e.g. ``'gray'``, ``'inferno'``,
        ``'viridis'``) → sum λ and display with that colormap. Default ``'gray'``.
    show_markers : bool - draw a circle (or rectangle) per source. Default True.
    show_ids : bool     - annotate each marker with its source index. Default False.
    show_weights : bool - fill markers with alpha ∝ source weight. Default False.
    roi : tuple of slices or None
        Region of interest, e.g. ``np.s_[..., y0:y1, x0:x1]``.
    title : str or None
    figsize : tuple   (used only when ax=None)
    norm : matplotlib Normalize or None
        Pre-built norm for cmap display; overrides scale/vmin/vmax when set.
    vmin, vmax : float or None  - cmap clipping bounds (ignored when norm is set)
    scale : {'log', 'linear'}
        Intensity scale for both cmap mode (auto-norm) and RGB mode. Default 'log'.
    saturation, contrast, wb_shift, mg_shift : float  - RGB colour adjustments
    marker_shape : {'circle', 'box'}
    marker_color : str          - any Matplotlib colour spec, default 'gold'
    marker_size : float or None - fixed radius [pixels]; None → log-scaled by flux
    min_marker_size, max_marker_size : float  - log-scaled radius range [pixels]
    show : bool   (used only when ax=None)
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # ── Local helpers (closures over DisplayField parameters) ─────────────────
    def _roi_offsets(roi_):
        if roi_ is None:
            return 0, 0
        slices = [s for s in roi_ if isinstance(s, slice)]
        if len(slices) >= 2:
            return (slices[-1].start or 0), (slices[-2].start or 0)
        return 0, 0

    def _draw_source_markers(ax_, src_df_, x_offset=0, y_offset=0):
        """Draw per-source markers, capturing style params from outer scope."""
        if src_df_ is None or len(src_df_) == 0:
            return
        xs   = src_df_['x_peak'].to_numpy() - x_offset
        ys   = src_df_['y_peak'].to_numpy() - y_offset
        peak = np.clip(src_df_['peak_value'].to_numpy().astype(float), 1e-10, None)
        wts  = src_df_['weight'].to_numpy().astype(float) if 'weight' in src_df_.columns else np.ones(len(xs))

        if marker_size is not None:
            radii = np.full(len(xs), float(marker_size))
        else:
            log_p = np.log10(peak)
            span  = log_p.max() - log_p.min()
            radii = min_marker_size + (log_p - log_p.min()) / (span + 1e-10) * (max_marker_size - min_marker_size)

        base       = np.array(mcolors.to_rgba(marker_color))
        edge_color = (*base[:3], 0.9)

        for i, (x, y, r) in enumerate(zip(xs, ys, radii)):
            fill_alpha = float(np.clip(wts[i], 0, 1)) * 0.75 if show_weights else 0.0
            
            if marker_shape == 'circle':
                if fill_alpha > 0:
                    ax_.add_patch(mpatches.Circle((x, y), r, facecolor=(*base[:3], fill_alpha), edgecolor='none', zorder=8))
                ax_.add_patch(mpatches.Circle((x, y), r, fill=False, edgecolor=edge_color, linewidth=0.8, zorder=9))
            else:  # 'box'
                side = 2 * r
                if fill_alpha > 0:
                    ax_.add_patch(mpatches.Rectangle((x - r, y - r), side, side, facecolor=(*base[:3], fill_alpha), edgecolor='none', zorder=8))
                ax_.add_patch(mpatches.Rectangle((x - r, y - r), side, side, fill=False, edgecolor=edge_color, linewidth=0.8, zorder=9))
            
            if show_ids:
                ax_.text(x + r * 0.8, y + r * 0.8, str(src_df_.index[i]), color=marker_color, fontsize=6, va='bottom', ha='left', zorder=11)

    def _get_sources_table(sources):
        """Normalise SourcesData / SourcesSubset / DataFrame to a DataFrame (or None)."""
        if sources is None:
            return None

        return sources.table if hasattr(sources, 'table') else sources

    # ── Prepare array ──────────────────────────────────────────────────────────
    arr = cube.detach().cpu().numpy() if isinstance(cube, torch.Tensor) else np.asarray(cube)
    arr = np.abs(arr)
    if roi is not None:
        arr = arr[roi]

    # ── Figure / axes ──────────────────────────────────────────────────────────
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=figsize)

    # ── Render image ───────────────────────────────────────────────────────────
    if cmap is None and arr.ndim == 3:
        # RGB mode: wavelength -> colour mapping
        # vmin/vmax override min_val/max_val for a unified intensity interface
        display_img, _, _ = _cube_to_RGB_array(
            arr,
            wavelengths,
            vmin,
            vmax,
            scale,
            saturation,
            contrast,
            wb_shift,
            mg_shift
        )
        ax.imshow(display_img, origin='lower')
        
    else:
        # cmap mode: collapse λ and display with chosen colormap
        img_2d = arr.sum(axis=0) if arr.ndim == 3 else arr
        img_2d = np.nan_to_num(img_2d)
        
        if norm is None:
            pos  = img_2d[img_2d > 0]
            v_lo = float(vmin if vmin is not None else (max(1.0, pos.min()) if len(pos) else 1.0))
            v_hi = float(vmax if vmax is not None else max(img_2d.max(), v_lo * 2))
            if scale == 'log':
                norm = LogNorm(vmin=v_lo, vmax=v_hi)
            else:
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=v_lo, vmax=v_hi)
        
        ax.imshow(img_2d, norm=norm, origin='lower', cmap=cmap if cmap is not None else 'gray')

    ax.axis('off')
    if title:
        ax.set_title(title)

    # ── Source markers ─────────────────────────────────────────────────────────
    if show_markers:
        src_df = _get_sources_table(sources)
        if src_df is not None and len(src_df):
            x_off, y_off = _roi_offsets(roi)
            _draw_source_markers(ax, src_df, x_offset=x_off, y_offset=y_off)

    if own_figure:
        plt.tight_layout()
        if show:
            plt.show()

    return ax


def DetectSources(data_cube, threshold=None, nsigma=3.0, box_size=11, sort_by_brightness=True, weight_from_flux=False, display=False, draw_box_size=None, verbose=False):
    """ Detects sources in a 3D data cube. """
    
    def _detect_sources(data_src, threshold, box_size, eps=2, verbose=False):
        sources = find_peaks(data_src, threshold=threshold, box_size=box_size)

        if sources is None:
            if verbose:
                print("No sources detected.")
            return pd.DataFrame(columns=['x_peak', 'y_peak', 'peak_value'])

        if verbose:
            print(f"Detected {len(sources)} sources")

        # Helps in the case if a single source was detected as multiple
        positions = np.transpose((sources['x_peak'], sources['y_peak']))
        
        db = DBSCAN(eps=eps, min_samples=1).fit(positions)

        unique_labels = set(db.labels_)
        merged_positions = np.array([ positions[db.labels_ == label].mean(axis=0) for label in unique_labels ])
        merged_fluxes    = np.array([ data_src[int(pos[1]), int(pos[0])] for pos in merged_positions ])

        merged_sources = pd.DataFrame(merged_positions, columns=['x_peak', 'y_peak'])
        merged_sources['peak_value'] = merged_fluxes

        if verbose:
            print(f"Merged to {len(merged_sources)} sources")
        
        return merged_sources

    def _auto_threshold(data_src, nsigma):
        """
        Estimate threshold from the image histogram using the mirrored-noise estimator.
        Only sub-background pixels are used to measure noise sigma, so the estimate is
        unbiased by source flux even in crowded fields.
        """
        flat = data_src.ravel()
        flat = flat[np.isfinite(flat) & (flat > 0)]
        
        if len(flat) == 0:
            return 0.0
        
        _, bg_level, _ = sigma_clipped_stats(flat, sigma=nsigma)
        below_bg = flat[flat < bg_level]
        
        if len(below_bg) > 10:
            # Reflect sub-background pixels around bg_level for an unbiased noise sigma
            noise_sigma = np.std(np.concatenate([below_bg, 2.0 * bg_level - below_bg]))
        else:
            _, bg_level, noise_sigma = sigma_clipped_stats(flat, sigma=nsigma)
            
        return bg_level + nsigma * noise_sigma

    data_src = data_cube.sum(dim=0).cpu().numpy() if isinstance(data_cube, torch.Tensor) else data_cube.sum(axis=0)
    data_src = np.nan_to_num(data_src)

    if threshold is None or threshold == 'auto':
        threshold = _auto_threshold(data_src, nsigma)
        if verbose:
            print(f"[auto] threshold from histogram = {threshold:.2f}")

    sources_df = _detect_sources(data_src, threshold=threshold, box_size=box_size, verbose=verbose)

    if len(sources_df) == 0:
        if verbose:
            print("No sources detected. Change threshold or nsigma to detect sources.")
        return sources_df

    # Draw the detected sources
    if display and draw_box_size is not None:
        draw_box_size = box_size if draw_box_size is None else int(draw_box_size)
        bg = float(np.nanmedian(data_src[data_src > 0]))
        DisplayField(data_src, sources=sources_df, show_markers=True, show_ids=True,
                     marker_shape='box', marker_size=draw_box_size/2,
                     vmin=bg*0.9, vmax=threshold*10)

    # Sort in descending order of flux (peak value)
    if sort_by_brightness:
        sources_df = sources_df.sort_values(by='peak_value', ascending=False).reset_index(drop=True)

    # Add weight column
    if weight_from_flux:
        # In the case there is a goal to assign more weight to brighter sources
        sources_df['weight'] = sources_df['peak_value'] / sources_df['peak_value'].sum()
    else:
        sources_df['weight'] = 1.0

    sources_df.index.name = 'ID'
    return sources_df


def AddSources(data_cube, coords, sources_df=None, weights=None, weight_from_flux=False):
    """
    Append manually specified sources to an existing (or new) sources DataFrame.

    Parameters
    ----------
    data_cube : torch.Tensor or np.ndarray
        3D data cube (C, H, W) used to measure peak values at the given coordinates.
    coords : array-like of shape (N, 2)
        Pixel coordinates [[x0, y0], [x1, y1], ...] for the sources to add.
    sources_df : pd.DataFrame or None
        Existing sources DataFrame with columns [x_peak, y_peak, peak_value, weight].
        If None, a new one is created.
    weights : array-like of length N or None
        Per-source weights. If None and weight_from_flux=False, weight=1.0.
        If None and weight_from_flux=True, weight is proportional to peak_value.
    weight_from_flux : bool
        When True and weights is None, assign weight = peak_value / total_peak_value
        across all sources in the returned DataFrame (including any pre-existing ones).

    Returns
    -------
    pd.DataFrame
        Updated (or newly created) sources DataFrame.
    """
    data_np = data_cube.sum(dim=0).cpu().numpy() if isinstance(data_cube, torch.Tensor) else data_cube.sum(axis=0)
    data_np = np.nan_to_num(data_np)

    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    N = len(coords)
    if weights is not None:
        weights = np.broadcast_to(np.asarray(weights, dtype=float).ravel(), (N,))

    rows = []
    for i, (x, y) in enumerate(coords):
        xi, yi = int(round(x)), int(round(y))
        xi = np.clip(xi, 0, data_np.shape[1] - 1)
        yi = np.clip(yi, 0, data_np.shape[0] - 1)
        peak_value = float(data_np[yi, xi])

        w = float(weights[i]) if weights is not None else 1.0
        rows.append({'x_peak': x, 'y_peak': y, 'peak_value': peak_value, 'weight': w})

    new_df = pd.DataFrame(rows)

    if sources_df is not None:
        result_df = pd.concat([sources_df, new_df], ignore_index=True)
    else:
        result_df = new_df.reset_index(drop=True)

    if weights is None and weight_from_flux:
        total = result_df['peak_value'].sum()
        result_df['weight'] = result_df['peak_value'] / total if total > 0 else 1.0

    result_df.index.name = 'ID'
    return result_df


def ExtractSourceImages(data_cube, srcs_coords, box_size, filter_sources=True, max_nan_fraction=0.3, debug_draw=False):
    ROIs, local_coords, global_coords, valid_srcs = extract_ROIs(data_cube, srcs_coords, box_size=box_size, max_nan_fraction=max_nan_fraction)
    sources_valid = srcs_coords.iloc[valid_srcs] if filter_sources else srcs_coords

    if debug_draw:
        N_cols = min(8, int(np.ceil(np.sqrt(len(ROIs))))) # Automatically adjusts the number of displayed columns
        plot_ROIs_as_grid(ROIs, cols=N_cols)

    # The split between local and global coordinates is necessary because PSF model simulates PSFs of the same size, while an object
    # Can be only partially present within the field. Then, simulated PSF is first cropped locally and then placed globally back into the full image
    return {
        "src_images": ROIs,          # Sources images extracted from the original image, with NaN padding if the source is close to the edge
        "src_data":   sources_valid, # DataFrame with source coordinates and other properties, filtered to only include sources with valid ROIs (if filter_sources=True)
        "ROI_global": global_coords, # List of tuples with global image coordinates (slices) corresponding to each ROI, used for placing the simulated PSF back into the full image
        "ROI_local":  local_coords   # List of tuples with local image coordinates (slices) corresponding to each ROI
    }


def extract_ROIs(image, sources, box_size=20, max_nan_fraction=0.3):
    torch_flag = False
    if isinstance(image, np.ndarray):
        xp = np
    elif isinstance(image, torch.Tensor):
        xp = torch
        torch_flag = True
    else:
        raise TypeError("Unexpected image data type")
        
    ROIs = []
    roi_local_coords  = []  # To store the local image indexes inside NaN-padded ROI
    roi_global_coords = []  # To store the coordinates relative to the original image
    positions = np.transpose((sources['x_peak'], sources['y_peak']))

    D = image.shape[0]  # Depth dimension

    half_box    = box_size // 2
    extra_pixel = box_size % 2  # 1 if box_size is odd, 0 if even

    valid_ids = []

    for i,pos in enumerate(positions):
        x, y = int(pos[0]), int(pos[1])
        
        # Calculate the boundaries, ensuring they don't exceed the image size
        x_min = x - half_box
        x_max = x + half_box + extra_pixel
        y_min = y - half_box
        y_max = y + half_box + extra_pixel
        
        # Extract the ROI with NaN-padding if the ROI goes outside the image bounds
        if torch_flag:
            roi = torch.full((D, box_size, box_size), float('nan'), device=image.device)  # Create a blank 3D box filled with NaNs
        else:
            roi = xp.full((D, box_size, box_size), float('nan'))  # Create a blank 3D box filled with NaNs
        
        # Calculate the actual overlapping region between image and ROI
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, image.shape[-1])
        y_max_img = min(y_max, image.shape[-2])

        # Determine where the image will go in the NaN-padded ROI
        x_min_roi = max(0, -x_min)
        y_min_roi = max(0, -y_min)
        x_max_roi = x_min_roi + (x_max_img - x_min_img)
        y_max_roi = y_min_roi + (y_max_img - y_min_img)

        # Copy image data into the padded ROI
        roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi] = image[:, y_min_img:y_max_img, x_min_img:x_max_img]
        
        # Filter out ROIs with too many NaNs
        if torch_flag:
            nan_fraction = torch.isnan(roi).sum().item() / roi.numel()
        else:
            nan_fraction = xp.isnan(roi).sum() / roi.size
            
        if nan_fraction <= max_nan_fraction:
            ROIs.append(roi)
            # Store the local coordinates where the actual image data is inside the NaN-padded ROI
            roi_local_coords.append(((y_min_roi, y_max_roi), (x_min_roi, x_max_roi)))
            # Store the global coordinates relative to the original image
            roi_global_coords.append(((y_min_img, y_max_img), (x_min_img, x_max_img)))
    
            valid_ids.append(i)
    
    return ROIs, roi_local_coords, roi_global_coords, valid_ids


def extract_ROIs_from_coords(image, roi_local_coords, roi_global_coords, PSF_size):
    torch_flag = False
    if isinstance(image, np.ndarray):
        xp = np
    elif isinstance(image, torch.Tensor):
        xp = torch
        torch_flag = True
    else:
        raise TypeError("Unexpected image data type")
        
    ROIs = []
    D = image.shape[0]  # Depth dimension

    for local_coord, global_coord in zip(roi_local_coords, roi_global_coords):
        (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coord
        (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coord

        # Create a blank 3D box filled with zeros
        if torch_flag:
            ROI = torch.zeros((D, PSF_size, PSF_size), dtype=image.dtype, device=image.device)
        else:
            ROI = xp.zeros((D, PSF_size, PSF_size), dtype=image.dtype, device=image.device)
        
        ROI[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi] = image[:, y_min_img:y_max_img, x_min_img:x_max_img]
        ROIs.append(ROI)

    return ROIs


def add_ROIs(image, ROIs, local_coords, global_coords):    
    # If ROIs is a torch tensor with ROIs in the 0th dimension
    if isinstance(ROIs, torch.Tensor) and ROIs.ndim == 4:  # [num_rois, channels, height, width]
        for i in range(ROIs.shape[0]):
            roi = ROIs[i]
        
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coords[i]
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coords[i]

            image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:       
        # Original implementation for list of ROIs
        for roi, local_idx, global_idx in zip(ROIs, local_coords, global_coords):
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx
    
            image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    return image


def add_ROIs_separately(image, ROIs, local_coords, global_coords):
    # If ROIs is a torch tensor with ROIs in the 0th dimension
    
    # Repeat image for each ROI
    if isinstance(image, torch.Tensor):
        images = [image.clone() for _ in range(len(ROIs))]
    else:
        images = [image.copy() for _ in range(len(ROIs))]
    
    if isinstance(ROIs, torch.Tensor) and ROIs.ndim == 4:  # [num_rois, channels, height, width]
        for i in range(ROIs.shape[0]):
            roi = ROIs[i]
        
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coords[i]
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coords[i]

            images[i][:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:
        # Original implementation for list of ROIs
        for i, (roi, local_idx, global_idx) in enumerate(zip(ROIs, local_coords, global_coords)):
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx
    
            images[i][:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    return torch.stack(images, dim=0)


def plot_ROIs_as_grid(ROIs, cols=5):
    """Display the ROIs in a grid of subplots."""
    n_ROIs = len(ROIs)
    rows = (n_ROIs + cols - 1) // cols  # Calculate number of rows needed
    _, axes = plt.subplots(rows, cols) #, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, roi in enumerate(ROIs):
        ax = axes[i]
        roi_ = roi.cpu() if roi.ndim == 2 else roi.sum(dim=0).cpu()  # Remove the channel dimension if it exists
        norm = simple_norm(roi_, 'log', percent=100-1e-1)
        ax.imshow(roi_, origin='lower', cmap='gray', norm=norm)
        # ax.set_title(f'Source {i+1}')
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # plt.tight_layout()
    plt.show()
    

def PlotSourcesProfiles(data, model, sources, radius, select_ids=None, show=True, **kwargs):
    # Extract PSFs from the image
    box_size = np.round(radius * 2 + 4).astype(int)

    ROIs_0, _, _, _ = extract_ROIs(data,  sources, box_size=box_size)
    ROIs_1, _, _, _ = extract_ROIs(model, sources, box_size=box_size)

    if select_ids is not None:
        ROIs_0 = [roi for i, roi in enumerate(ROIs_0) if i in select_ids]
        ROIs_1 = [roi for i, roi in enumerate(ROIs_1) if i in select_ids]

    # Spectrally average the PSFs
    avg_white = (
        lambda x: torch.stack(x).mean(dim=1)
        if isinstance(x[0], torch.Tensor)
        else np.mean(np.stack(x), axis=1)
    )

    PSFs_0_white = avg_white(ROIs_0)
    PSFs_1_white = avg_white(ROIs_1)

    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.gca()

    plot_radial_PSF_profiles(
        PSF_0 = PSFs_0_white,
        PSF_1 = PSFs_1_white,
        label_0 = 'Data',
        label_1 = 'Prediction',
        cutoff=radius,
        ax=ax,
        **kwargs,
    )
    if show:
        plt.show()


def select_sources(src_dict: dict, selected_ids: Union[list, int]) -> dict:
    if isinstance(selected_ids, int):
        selected_ids = [selected_ids]
    result_dict = {}
    for key, tensor in src_dict.items():
        if hasattr(tensor, 'shape') and tensor.shape[0] > 1:
            result_dict[key] = tensor[selected_ids]
        else:
            result_dict[key] = tensor
    return result_dict


def ROI_from_valid_mask(
    mask,
    *,
    fill_holes=True,
    close_radius=1,
    occupancy_threshold=0.999,
    center=None,
    return_clean_mask=True,
):
    """
    Find the largest axis-aligned square grown from a given center inside a binary mask.

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor
        2D binary/grayscale mask. Nonzero = inside.
    fill_holes : bool
        Fill internal black holes in the white region.
    close_radius : int
        Radius of morphological closing to suppress tiny black defects/cracks.
        Set to 0 to disable.
    occupancy_threshold : float
        Fraction of white pixels required inside the candidate square.
        1.0 means strict containment.
        Slightly below 1 tolerates a few bad black pixels.
    center : tuple[float, float] or None
        (cx, cy) in pixel coordinates. If None, uses mask centroid.
    return_clean_mask : bool
        Whether to include the cleaned mask in the output.

    Returns
    -------
    out : dict
        {
            "center": (cx, cy),
            "half_size": ...,
            "side_pixels": ...,
            "bbox": (x0, y0, x1, y1),   # x1,y1 exclusive
            "corners": array of shape (4, 2),
            "slice": np.s_[..., y0:y1, x0:x1],
            "mask_clean": ...           # only if return_clean_mask=True
        }
    """
    # Torch-safe conversion without importing torch
    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(np.asarray(mask))
    
    if mask.ndim != 2:
        raise ValueError(f"`mask` must be 2D, got shape {mask.shape}")

    # --- clean mask ---
    mask_clean = mask > 0

    if fill_holes:
        mask_clean = ndimage.binary_fill_holes(mask_clean)

    if close_radius > 0:
        y, x = np.ogrid[-close_radius:close_radius+1, -close_radius:close_radius+1]
        selem = (x * x + y * y) <= close_radius * close_radius
        mask_clean = ndimage.binary_closing(mask_clean, structure=selem)

    H, W = mask_clean.shape

    # --- choose center ---
    if center is None:
        ys, xs = np.nonzero(mask_clean)
        if len(xs) == 0:
            raise ValueError("Empty mask after cleaning.")
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = center

    # --- integral image ---
    ii = np.pad(mask_clean.astype(np.uint8), ((1, 0), (1, 0)), mode="constant")
    ii = ii.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(x0, y0, x1, y1):
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

    def fits(half_size):
        x0 = int(np.ceil(cx - half_size))
        x1 = int(np.floor(cx + half_size)) + 1
        y0 = int(np.ceil(cy - half_size))
        y1 = int(np.floor(cy + half_size)) + 1

        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            return False

        area = (x1 - x0) * (y1 - y0)
        inside = rect_sum(x0, y0, x1, y1)
        return inside >= occupancy_threshold * area

    # --- binary search for largest square ---
    lo, hi = 0.0, min(H, W) / 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            lo = mid
        else:
            hi = mid

    half_size = lo
    x0 = int(np.ceil(cx - half_size))
    x1 = int(np.floor(cx + half_size)) + 1
    y0 = int(np.ceil(cy - half_size))
    y1 = int(np.floor(cy + half_size)) + 1

    bbox = (x0, y0, x1, y1)
    corners = np.array([
        [x0,     y0],
        [x1 - 1, y0],
        [x1 - 1, y1 - 1],
        [x0,     y1 - 1],
    ], dtype=int)

    out = {
        "center": (cx, cy),
        "half_size": half_size,
        "side_pixels": min(x1 - x0, y1 - y0),
        "bbox": bbox,
        "corners": corners,
        "slice": np.s_[..., y0:y1, x0:x1],
    }

    if return_clean_mask:
        out["mask_clean"] = mask_clean

    return out
