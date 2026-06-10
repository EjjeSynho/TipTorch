from __future__ import annotations

import re
import math
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.eso import Eso


# -----------------------------------------------------------------------------
# FITS / metadata extraction
# -----------------------------------------------------------------------------

def _headers_from_fits(path: str | Path) -> list[fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        return [hdu.header.copy() for hdu in hdul]


def _get_first(headers: Sequence[fits.Header], keys: Iterable[str], default=None):
    for hdr in headers:
        for key in keys:
            if key in hdr:
                value = hdr.get(key)
                if value not in (None, "", "UNKNOWN", "None"):
                    return value
    return default


def _normalise_dp_id(value) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    text = text.replace(".fits.fz", "").replace(".fits", "").replace(".fz", "")

    m = re.search(
        r"\b(?:MUSE|ADP)\.\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\b",
        text,
    )
    return m.group(0) if m else None


def _extract_muse_dp_ids_from_text(text: str) -> list[str]:
    return sorted(set(re.findall(
        r"\bMUSE\.\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\b",
        str(text),
    )))


def extract_reduced_muse_cube_metadata(cube_path: str | Path) -> dict:
    """
    Extract matching metadata from a reduced MUSE-NFM cube.

    This deliberately checks many possible keyword variants because reduced MUSE
    products can come from different ESO pipeline / Phase-3 products.
    """
    headers = _headers_from_fits(cube_path)

    ra = _get_first(headers, ["RA", "CRVAL1"])
    dec = _get_first(headers, ["DEC", "CRVAL2"])

    coord = None
    if ra is not None and dec is not None:
        try:
            coord = SkyCoord(float(ra), float(dec), unit="deg")
        except Exception:
            coord = None

    date_obs = _get_first(headers, ["DATE-OBS", "DATE"])
    mjd_obs = _get_first(headers, ["MJD-OBS"])
    mjd_end = _get_first(headers, ["MJD-END", "MJD-END "])

    if date_obs is None and mjd_obs is not None:
        date_obs = Time(float(mjd_obs), format="mjd").isot

    exp_start = date_obs
    exp_end = None

    if mjd_end is not None:
        exp_end = Time(float(mjd_end), format="mjd").isot
    else:
        exptime = _get_first(headers, ["EXPTIME", "HIERARCH ESO DET SEQ1 EXPTIME"])
        if date_obs is not None and exptime is not None:
            try:
                exp_end = (Time(date_obs) + float(exptime) * u.s).isot
            except Exception:
                pass

    # Common ESO hierarchy keywords.
    prog_id = _get_first(headers, [
        "HIERARCH ESO OBS PROG ID",
        "ESO OBS PROG ID",
        "PROG_ID",
        "PROGID",
        "PROGID1",
    ])

    obs_id = _get_first(headers, [
        "HIERARCH ESO OBS ID",
        "ESO OBS ID",
        "OBS.ID",
        "OBID1",
        "OB_ID",
    ])

    tpl_start = _get_first(headers, [
        "HIERARCH ESO TPL START",
        "ESO TPL START",
        "TPL.START",
        "TPLSTART",
    ])

    tpl_id = _get_first(headers, [
        "HIERARCH ESO TPL ID",
        "ESO TPL ID",
        "TPL.ID",
        "TPLID",
    ])

    object_name = _get_first(headers, [
        "OBJECT",
        "HIERARCH ESO OBS TARG NAME",
        "ESO OBS TARG NAME",
    ])

    ins_mode = _get_first(headers, [
        "HIERARCH ESO INS MODE",
        "ESO INS MODE",
        "INS.MODE",
    ])

    dpr_catg = _get_first(headers, [
        "HIERARCH ESO DPR CATG",
        "ESO DPR CATG",
        "DPR.CATG",
    ])

    dpr_type = _get_first(headers, [
        "HIERARCH ESO DPR TYPE",
        "ESO DPR TYPE",
        "DPR.TYPE",
    ])

    reduced_dp_id = _normalise_dp_id(
        _get_first(headers, ["DP.ID", "ARCFILE", "ORIGFILE"])
    )

    # Direct provenance: strongest possible signal.
    provenance_raw_dp_ids: list[str] = []
    for hdr in headers:
        for key, value in hdr.items():
            key_upper = str(key).upper()
            if (
                key_upper.startswith("PROV")
                or key_upper.startswith("RAW")
                or key_upper.startswith("ASSOC")
                or key_upper in {"ORIGFILE", "ARCFILE"}
                or key_upper in {"HISTORY", "COMMENT"}
            ):
                provenance_raw_dp_ids.extend(_extract_muse_dp_ids_from_text(str(value)))

    provenance_raw_dp_ids = sorted(set(provenance_raw_dp_ids))

    return {
        "cube_path": str(cube_path),
        "reduced_dp_id": reduced_dp_id,
        "ra_deg": None if coord is None else coord.ra.deg,
        "dec_deg": None if coord is None else coord.dec.deg,
        "exp_start": exp_start,
        "exp_end": exp_end,
        "mjd_obs": None if mjd_obs is None else float(mjd_obs),
        "mjd_end": None if mjd_end is None else float(mjd_end),
        "object": object_name,
        "prog_id": prog_id,
        "obs_id": None if obs_id is None else str(obs_id),
        "tpl_start": tpl_start,
        "tpl_id": tpl_id,
        "ins_mode": ins_mode,
        "dpr_catg": dpr_catg,
        "dpr_type": dpr_type,
        "provenance_raw_dp_ids": provenance_raw_dp_ids,
    }


# -----------------------------------------------------------------------------
# ESO archive querying, public-only
# -----------------------------------------------------------------------------

def query_public_muse_raw_candidates(
    cube_meta: dict,
    *,
    time_padding_hours: float = 12.0,
    radius_arcmin: float = 1.5,
    row_limit: Optional[int] = None,
) -> Table:
    """
    Query public MUSE raw candidates from ESO.

    This is anonymous/public-only: no login, no authenticated=True.

    Parameters
    ----------
    cube_meta
        Output of extract_reduced_muse_cube_metadata().
    time_padding_hours
        Search window around reduced cube exp_start/exp_end.
    radius_arcmin
        Cone-search radius around cube coordinates.
    row_limit
        ESO row limit. None means no limit.
    """
    eso = Eso()
    eso.ROW_LIMIT = row_limit if row_limit is not None else -1

    filters = {}

    # Query around exposure start/end. ESO TAP uses exp_start rather than old WDB
    # stime/etime fields.
    if cube_meta.get("exp_start") is not None:
        t0 = Time(cube_meta["exp_start"]) - time_padding_hours * u.hour

        if cube_meta.get("exp_end") is not None:
            t1 = Time(cube_meta["exp_end"]) + time_padding_hours * u.hour
        else:
            t1 = Time(cube_meta["exp_start"]) + time_padding_hours * u.hour

        filters["exp_start"] = f"between '{t0.isot}' and '{t1.isot}'"

    # Program ID is very useful when present, but do not make the query too brittle
    # if the reduced cube lacks it.
    if cube_meta.get("prog_id"):
        filters["prog_id"] = f"= '{cube_meta['prog_id']}'"

    columns = [
        "dp_id",
        "date_obs",
        "exp_start",
        "exptime",
        "object",
        "ra",
        "dec",
        "prog_id",
        "dp_cat",
        "dp_type",
        "dp_tech",
        "tpl_id",
        "tpl_start",
        "tpl_expno",
        "tpl_nexp",
        "ob_id",
        "release_date",
        "access_estsize",
    ]

    cone_kwargs = {}
    if cube_meta.get("ra_deg") is not None and cube_meta.get("dec_deg") is not None:
        cone_kwargs = {
            "cone_ra": float(cube_meta["ra_deg"]),
            "cone_dec": float(cube_meta["dec_deg"]),
            "cone_radius": float(radius_arcmin / 60.0),  # degrees
        }

    try:
        tbl = eso.query_instrument(
            "muse",
            columns=columns,
            column_filters=filters,
            **cone_kwargs,
            authenticated=False,  # important: public-only
        )
    except Exception:
        # Some astroquery/ESO combinations may not expose all columns.
        tbl = eso.query_instrument(
            "muse",
            columns=[
                "dp_id",
                "date_obs",
                "exp_start",
                "exptime",
                "object",
                "ra",
                "dec",
                "prog_id",
                "dp_cat",
                "dp_type",
                "tpl_id",
                "tpl_start",
                "ob_id",
                "release_date",
            ],
            column_filters=filters,
            **cone_kwargs,
            authenticated=False,
        )

    return Table() if tbl is None else tbl


def fetch_public_raw_headers(candidate_dp_ids: Sequence[str]) -> Table:
    """
    Fetch full ESO archive FITS headers for public candidate raw exposures.

    get_headers() does not download raw FITS data; it retrieves extended header
    metadata.
    """
    ids = sorted({_normalise_dp_id(x) for x in candidate_dp_ids if _normalise_dp_id(x)})
    if not ids:
        return Table()

    eso = Eso()
    eso.ROW_LIMIT = -1
    return eso.get_headers(ids)


# -----------------------------------------------------------------------------
# Matching / scoring
# -----------------------------------------------------------------------------

def _table_to_dataframe(tbl: Table) -> pd.DataFrame:
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame()
    return tbl.to_pandas()


def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "none", "null", "--"} else None


def _row_get(row: pd.Series, names: Sequence[str]):
    """
    Case-insensitive, ESO-header-friendly row getter.

    Handles both TAP columns like 'dp_id' and get_headers columns like
    'DP.ID' or 'HIERARCH ESO OBS ID'.
    """
    lower_map = {str(k).lower(): k for k in row.index}

    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            value = row[key]
            value = _safe_str(value)
            if value is not None:
                return value

    return None


def _to_time(value) -> Optional[Time]:
    value = _safe_str(value)
    if value is None:
        return None

    try:
        return Time(value)
    except Exception:
        pass

    try:
        return Time(float(value), format="mjd")
    except Exception:
        return None


def _time_abs_seconds(a, b) -> Optional[float]:
    ta = _to_time(a)
    tb = _to_time(b)
    if ta is None or tb is None:
        return None
    return abs((ta - tb).to_value(u.s))


def _coord_sep_arcsec(meta: dict, row: pd.Series) -> Optional[float]:
    if meta.get("ra_deg") is None or meta.get("dec_deg") is None:
        return None

    ra = _row_get(row, ["ra", "RA"])
    dec = _row_get(row, ["dec", "DEC"])

    if ra is None or dec is None:
        return None

    try:
        c0 = SkyCoord(float(meta["ra_deg"]), float(meta["dec_deg"]), unit="deg")
        c1 = SkyCoord(float(ra), float(dec), unit="deg")
        return c0.separation(c1).to_value(u.arcsec)
    except Exception:
        return None


def _normalised_text_equal(a, b) -> bool:
    a = _safe_str(a)
    b = _safe_str(b)
    if a is None or b is None:
        return False

    def norm(s):
        return re.sub(r"\s+", "", s).upper()

    return norm(a) == norm(b)


def _contains_nfm(*values) -> bool:
    text = " ".join(str(v) for v in values if v is not None).upper()
    return "NFM" in text


def score_candidate_against_cube(meta: dict, row: pd.Series) -> dict:
    """
    Assign a heuristic confidence score to one raw candidate.

    The score is intentionally transparent: each matching field contributes
    named points, so you can inspect why a candidate was accepted/rejected.
    """
    score = 0.0
    reasons: list[str] = []
    penalties: list[str] = []

    dp_id = _row_get(row, ["dp_id", "DP.ID"])

    # Strongest possible: raw DP.ID appears directly in reduced product provenance.
    if dp_id and dp_id in set(meta.get("provenance_raw_dp_ids", [])):
        score += 100
        reasons.append("raw DP.ID appears in reduced-cube provenance")

    # Program ID.
    prog_id = _row_get(row, ["prog_id", "HIERARCH ESO OBS PROG ID", "ESO OBS PROG ID"])
    if meta.get("prog_id") and prog_id:
        if _normalised_text_equal(meta["prog_id"], prog_id):
            score += 20
            reasons.append("program ID matches")
        else:
            score -= 25
            penalties.append(f"program ID differs: cube={meta['prog_id']} raw={prog_id}")

    # OB ID.
    ob_id = _row_get(row, ["ob_id", "OBS.ID", "HIERARCH ESO OBS ID", "ESO OBS ID"])
    if meta.get("obs_id") and ob_id:
        if _normalised_text_equal(meta["obs_id"], ob_id):
            score += 20
            reasons.append("OB ID matches")
        else:
            score -= 10
            penalties.append(f"OB ID differs: cube={meta['obs_id']} raw={ob_id}")

    # Template start.
    tpl_start = _row_get(row, ["tpl_start", "TPL.START", "HIERARCH ESO TPL START", "ESO TPL START"])
    if meta.get("tpl_start") and tpl_start:
        dt = _time_abs_seconds(meta["tpl_start"], tpl_start)
        if dt is not None and dt < 2:
            score += 25
            reasons.append("TPL.START matches")
        elif dt is not None and dt < 60:
            score += 12
            reasons.append(f"TPL.START nearly matches: Δ={dt:.1f}s")
        else:
            score -= 10
            penalties.append(f"TPL.START differs: Δ={dt:.1f}s" if dt is not None else "TPL.START differs")

    # Template ID.
    tpl_id = _row_get(row, ["tpl_id", "TPL.ID", "HIERARCH ESO TPL ID", "ESO TPL ID"])
    if meta.get("tpl_id") and tpl_id:
        if _normalised_text_equal(meta["tpl_id"], tpl_id):
            score += 10
            reasons.append("TPL.ID matches")

    # Exposure start.
    raw_start = _row_get(row, ["exp_start", "date_obs", "DATE-OBS", "MJD-OBS"])
    if meta.get("exp_start") and raw_start:
        dt = _time_abs_seconds(meta["exp_start"], raw_start)
        if dt is not None:
            if dt < 2:
                score += 25
                reasons.append("exposure start matches")
            elif dt < 120:
                score += 15
                reasons.append(f"exposure start close: Δ={dt:.1f}s")
            elif dt < 3600:
                score += 5
                reasons.append(f"exposure start within 1h: Δ={dt / 60:.1f}min")
            else:
                score -= min(30, dt / 3600)
                penalties.append(f"exposure start far: Δ={dt / 3600:.2f}h")

    # Exposure end, when the raw row has exptime.
    raw_exptime = _row_get(row, ["exptime", "EXPTIME"])
    if meta.get("exp_end") and raw_start and raw_exptime:
        try:
            raw_end = (Time(raw_start) + float(raw_exptime) * u.s).isot
            dt_end = _time_abs_seconds(meta["exp_end"], raw_end)
            if dt_end is not None:
                if dt_end < 2:
                    score += 20
                    reasons.append("exposure end matches")
                elif dt_end < 120:
                    score += 10
                    reasons.append(f"exposure end close: Δ={dt_end:.1f}s")
        except Exception:
            pass

    # Coordinates.
    sep_arcsec = _coord_sep_arcsec(meta, row)
    if sep_arcsec is not None:
        if sep_arcsec < 1:
            score += 20
            reasons.append(f"pointing matches: {sep_arcsec:.2f} arcsec")
        elif sep_arcsec < 10:
            score += 12
            reasons.append(f"pointing close: {sep_arcsec:.2f} arcsec")
        elif sep_arcsec < 60:
            score += 5
            reasons.append(f"pointing within 1 arcmin: {sep_arcsec:.1f} arcsec")
        else:
            score -= min(25, sep_arcsec / 60)
            penalties.append(f"pointing offset large: {sep_arcsec:.1f} arcsec")

    # Object name.
    raw_object = _row_get(row, ["object", "OBJECT"])
    if meta.get("object") and raw_object:
        if _normalised_text_equal(meta["object"], raw_object):
            score += 10
            reasons.append("object name matches")

    # MUSE NFM / AO mode sanity check.
    raw_dp_tech = _row_get(row, ["dp_tech", "DPR.TECH", "HIERARCH ESO DPR TECH"])
    raw_ins_mode = _row_get(row, ["INS.MODE", "HIERARCH ESO INS MODE", "ESO INS MODE"])

    cube_mentions_nfm = _contains_nfm(meta.get("ins_mode"), meta.get("dpr_type"), meta.get("dpr_catg"))
    raw_mentions_nfm = _contains_nfm(raw_dp_tech, raw_ins_mode, _row_get(row, ["dp_type", "DPR.TYPE"]))

    if cube_mentions_nfm and raw_mentions_nfm:
        score += 15
        reasons.append("both cube and raw candidate indicate NFM")
    elif cube_mentions_nfm and not raw_mentions_nfm:
        score -= 10
        penalties.append("cube appears NFM, raw candidate does not clearly indicate NFM")

    # DPR sanity: science object exposure preferred.
    dp_cat = _row_get(row, ["dp_cat", "DPR.CATG", "HIERARCH ESO DPR CATG"])
    dp_type = _row_get(row, ["dp_type", "DPR.TYPE", "HIERARCH ESO DPR TYPE"])

    dpr_text = f"{dp_cat or ''} {dp_type or ''}".upper()
    if "SCIENCE" in dpr_text:
        score += 8
        reasons.append("raw candidate is science category")
    if "OBJECT" in dpr_text:
        score += 8
        reasons.append("raw candidate is object exposure")
    if any(x in dpr_text for x in ["BIAS", "DARK", "FLAT", "ARC", "WAVE", "STD"]):
        score -= 20
        penalties.append(f"candidate may be calibration: {dpr_text.strip()}")

    # Publicness sanity: anonymous query should already return public results.
    release_date = _row_get(row, ["release_date", "RELEASE_DATE"])
    if release_date:
        try:
            if Time(release_date) <= Time.now():
                score += 3
                reasons.append("release_date is in the past")
            else:
                score -= 100
                penalties.append(f"not public yet: release_date={release_date}")
        except Exception:
            pass

    confidence = "low"
    if score >= 120:
        confidence = "very_high"
    elif score >= 80:
        confidence = "high"
    elif score >= 45:
        confidence = "medium"

    return {
        "dp_id": dp_id,
        "score": score,
        "confidence": confidence,
        "sep_arcsec": sep_arcsec,
        "reasons": "; ".join(reasons),
        "penalties": "; ".join(penalties),
    }


def match_public_muse_raws_to_reduced_cube(
    cube_path: str | Path,
    *,
    out_dir: str | Path = "eso_public_muse_raw_match",
    time_padding_hours: float = 12.0,
    radius_arcmin: float = 1.5,
    min_score: float = 45.0,
    fetch_full_headers: bool = True,
) -> dict:
    """
    Public-only reduced-cube -> raw-header association.

    Returns a ranked table of candidate raw frames and, optionally, their full
    ESO archive FITS headers.

    This does not download raw science data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = extract_reduced_muse_cube_metadata(cube_path)

    candidates = query_public_muse_raw_candidates(
        meta,
        time_padding_hours=time_padding_hours,
        radius_arcmin=radius_arcmin,
    )

    if len(candidates) == 0:
        raise RuntimeError("No public MUSE raw candidates found in the ESO archive query.")

    cand_df = _table_to_dataframe(candidates)

    # Score using the TAP candidate table first.
    score_rows = []
    for _, row in cand_df.iterrows():
        score_rows.append(score_candidate_against_cube(meta, row))

    score_df = pd.DataFrame(score_rows)

    ranked = cand_df.copy()
    ranked["_dp_id_key"] = ranked["dp_id"].astype(str)
    score_df["_dp_id_key"] = score_df["dp_id"].astype(str)

    ranked = ranked.merge(
        score_df.drop(columns=["dp_id"]),
        on="_dp_id_key",
        how="left",
    ).drop(columns=["_dp_id_key"])

    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    accepted = ranked[ranked["score"] >= min_score].copy()

    raw_headers = Table()
    if fetch_full_headers and len(accepted) > 0:
        raw_headers = fetch_public_raw_headers(accepted["dp_id"].tolist())

        # Re-score using full headers as well, because get_headers() contains
        # many fields missing from the TAP candidate row.
        if len(raw_headers) > 0:
            hdr_df = _table_to_dataframe(raw_headers)
            hdr_score_rows = []
            for _, row in hdr_df.iterrows():
                hdr_score_rows.append(score_candidate_against_cube(meta, row))

            hdr_scores = pd.DataFrame(hdr_score_rows).sort_values(
                "score", ascending=False
            )

            hdr_scores.to_csv(out_dir / "raw_header_scores.csv", index=False)

    # Save outputs.
    (out_dir / "reduced_cube_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    ranked.to_csv(out_dir / "ranked_public_raw_candidates.csv", index=False)
    accepted.to_csv(out_dir / "accepted_public_raw_candidates.csv", index=False)

    candidates.write(out_dir / "public_raw_candidates.ecsv", overwrite=True)
    if len(raw_headers) > 0:
        raw_headers.write(out_dir / "public_raw_headers.ecsv", overwrite=True)

    return {
        "cube_metadata": meta,
        "ranked_candidates": ranked,
        "accepted_candidates": accepted,
        "raw_headers": raw_headers,
        "out_dir": str(out_dir),
    }