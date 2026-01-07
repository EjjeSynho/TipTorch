#%%
from __future__ import annotations
import logging

import gdown
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import logging

from project_settings import TELEMETRY_CACHE

#%%
def create_bundles(
    overwrite: bool = False,
    compression: int = zipfile.ZIP_DEFLATED,
) -> list[Path]:
    """
    For each direct subdirectory of TELEMETRY_CACHE, create a ZIP bundle:

        reduced_telemetry/FOLDER/  ->  reduced_telemetry/FOLDER_bundle.zip

    Returns a list of created bundle paths.
    """
    created: list[Path] = []

    for folder in sorted(p for p in TELEMETRY_CACHE.iterdir() if p.is_dir()):
        bundle_path = folder.with_name(folder.name + "_bundle.zip")

        if bundle_path.exists() and not overwrite: continue

        with zipfile.ZipFile(bundle_path, "w", compression=compression) as zf:
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    # store path relative to the folder root
                    arcname = file_path.relative_to(folder)
                    zf.write(file_path, arcname)

        created.append(bundle_path)

    return created


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest() -> None:
    ''' Build a manifest file for the reduced telemetry data. '''
    local_root = TELEMETRY_CACHE
    ids_json = local_root / "file_ids.json"
    out_manifest = local_root / "manifest.json"

    ids = json.loads(Path(ids_json).read_text(encoding="utf-8"))

    items = []
    for p in sorted(local_root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(local_root).as_posix()

        if rel not in ids:
            # Skip files without a corresponding Drive ID instead of failing
            continue

        items.append(
            {
                "path": rel,
                "id": ids[rel],
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )

    manifest = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "items": items,
    }

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pack_reduced_telemetry_bundles() -> None:
    ''' Create ZIP bundles for all reduced telemetry folders and build manifest. '''
    ''' Resulting files are then has to be manually uploaded to Google Drive. '''
    local_ids_path = TELEMETRY_CACHE / "file_ids.json"

    try:
        gdown.download(id='1-dxEM7fRfUN_0gxdd6uG6jZRJUqHTBMd', output=str(local_ids_path), quiet=False)

        create_bundles(overwrite=True, compression=zipfile.ZIP_DEFLATED)
        build_manifest()
        logging.info("Successfully packed reduced telemetry bundles.")
        local_ids_path.unlink()
        
    except Exception as e:
        logging.exception("ERROR: Failed to pack reduced telemetry bundles.")
        raise e

pack_reduced_telemetry_bundles()

#%%
def download_from_manifest(
    manifest_file_id: str,
    overwrite: bool = False,
    verify_sha256: bool = True,
    quiet: bool = True,
) -> None:
    out_dir = TELEMETRY_CACHE
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) fetch manifest
    manifest_path = out_dir / "manifest.json"
    gdown.download(id=manifest_file_id, output=str(manifest_path), quiet=quiet)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])

    # 2) fetch each file
    for item in items:
        rel = item["path"]
        file_id = item["id"]
        expected_sha = item.get("sha256")

        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not overwrite:
            if verify_sha256 and expected_sha:
                if sha256_file(dest) == expected_sha:
                    continue  # up-to-date
            else:
                continue  # assume ok if no verification requested

        # Download to destination
        gdown.download(id=file_id, output=str(dest), quiet=quiet)

        if verify_sha256 and expected_sha:
            got = sha256_file(dest)
            if got != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {rel}: expected {expected_sha}, got {got}")

    # 3) Remove manifest file
    if manifest_path.exists():
        manifest_path.unlink()


#%%
def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """
    Safely extract a zip file into dest_dir, preventing path traversal (zip slip).
    """
    dest_dir = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = (dest_dir / member.filename).resolve()
            if not target_path.is_relative_to(dest_dir):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")

        zf.extractall(dest_dir)


def unpack_bundle_zips(
    root_dir: str | Path,
    recursive: bool = True,
    overwrite: bool = False,
    remove_bundle: bool = True,
) -> list[tuple[Path, Path]]:
    """
    Find '*_bundle.zip' files under root_dir, extract them into folders with
    '_bundle' removed from the name, and optionally delete the zip files.

    Example:
      foo/bar/MUSE_bundle.zip -> foo/bar/MUSE/
    """
    root_dir = TELEMETRY_CACHE

    globber = root_dir.rglob if recursive else root_dir.glob
    bundles = sorted(p for p in globber("*_bundle.zip") if p.is_file())

    results: list[tuple[Path, Path]] = []

    for zip_path in bundles:
        stem = zip_path.stem  # e.g. "MUSE_bundle"
        out_dir = zip_path.parent / stem[:-len("_bundle")]

        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            results.append((zip_path, out_dir))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        _safe_extract_zip(zip_path, out_dir) # Extract

        # Remove bundle only if extraction succeeded
        if remove_bundle: zip_path.unlink()

        results.append((zip_path, out_dir))

    return results


def fetch_reduced_telemetry_cache(verbose=False) -> None:
    """
    Download and unpack all reduced telemetry data bundles as per the manifest.
    """
    # Check if reduced telemetry folder exists. If not, create it
    if TELEMETRY_CACHE.exists() and any(TELEMETRY_CACHE.iterdir()):
        # Do nothing, files are present
        return
    
    try:
        logging.warning("Reduced telemetry data not found.")
        logging.info("Downloading reduced telemetry data...")
        
        TELEMETRY_CACHE.mkdir(parents=True, exist_ok=True)

        download_from_manifest(
            manifest_file_id='1KekiEHd9_4H6DKhEXky_bdIVfr3RfD5N',
            overwrite=True,
            verify_sha256=True,
            quiet=False,
        )

        # Extract all *_bundle.zip
        unpacked = unpack_bundle_zips(
            TELEMETRY_CACHE,
            recursive=True,
            overwrite=True,
            remove_bundle=True
        )

        for z, d in unpacked:
            print("Extracted", z, "->", d)
            
    except Exception as e:
        logging.exception("ERROR: Failed to fetch reduced telemetry data.")
        logging.info("Please try manually by visiting https://drive.google.com/drive/folders/1fZ6D9xsELHn-IHdA1Goy-QCmrOxCO6F0?usp=drive_link")
        raise e


# fetch_reduced_telemetry_cache()

# def DownloadFromGoogleDrive(file_id, output_path, overwrite=False, verbose=False):
#     """
#     Downloads a file from Google Drive using a shareable link.

#     Parameters:
#         file_id (str): ID of the file on Google Drive
#         output_path (str): Path where the file should be saved
#         overwrite (bool): If True, overwrites the file if it already exists.
#                           If False, skips download if file exists. Default is False.
#     """
#     # Check if the file exists and handle based on overwrite flag
#     if os.path.exists(output_path) and not overwrite:
#         print(f"File already exists at {output_path}. Set overwrite=True to replace it.")
#         return

#     share_url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'

#     os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create file's directory if it doesn't exist
#     gdown.download(share_url, output_path, quiet=not verbose, fuzzy=True) # Download the file
# %%
