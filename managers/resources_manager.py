#%%
import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import pooch

from project_settings import CACHE_PATH, RESOURCE_PACKS_DIR, TEMP_DIR, REGISTRY_URL

#%%
# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract *zip_path* into *dest*, rejecting entries that escape *dest* (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.infolist():
            if not (dest / m.filename).resolve().is_relative_to(dest):
                raise RuntimeError(f"Unsafe zip entry: {m.filename}")
        zf.extractall(dest)


def _parse_pack_list(txt_path: Path) -> list[str]:
    """
    Read a resource-pack .txt file and return normalised POSIX-style
    relative paths (e.g. ``calibrations/ELT/file.fits``).
    Blank lines are skipped.
    """
    lines: list[str] = []
    for raw in txt_path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Normalise Windows .\ and Unix ./ prefixes, then convert to forward slashes
        lines.append(PurePosixPath(Path(line)).as_posix().lstrip("./"))
    return lines


def _gdrive_download_url(url_or_id: str) -> str:
    """
    Accept either a bare GDrive file ID or a full sharing URL and return a
    direct-download URL.

    Supported input formats:
      - ``1YRs4WWmBY9EoGoAQqRCEdW1nKYU26Jwo``
      - ``https://drive.google.com/file/d/1YRs4WWmBY9EoGoAQqRCEdW1nKYU26Jwo/view?usp=drive_link``
      - ``https://drive.google.com/uc?export=download&id=1YRs4WWmBY9EoGoAQqRCEdW1nKYU26Jwo``
    """
    import re
    # Try to extract file ID from a /file/d/<ID> sharing link
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        file_id = m.group(1)
    # Try to extract from a ?id=<ID> direct link
    elif "id=" in url_or_id:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
        file_id = m.group(1) if m else url_or_id
    else:
        # Assume it's already a bare file ID
        file_id = url_or_id.strip()

    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ══════════════════════════════════════════════════════════════
#  DEVELOPER TOOLS - packing & registry management
# ══════════════════════════════════════════════════════════════

def pack_all(output_dir: Path | None = None, overwrite: bool = False) -> list[Path]:
    """
    Scan RESOURCE_PACKS_DIR for ``*.txt`` pack-list files.
    For each one, create a ZIP archive in *output_dir* (default: TEMP_DIR).

    Each archive contains:
      - every file listed in the ``.txt`` (stored under its relative path)
      - the ``.txt`` list itself (under ``resource_packs/<name>.txt``)

    Returns a list of created ZIP paths.
    """
    out = output_dir or TEMP_DIR
    out.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RESOURCE_PACKS_DIR.glob("*.txt"))
    if not txt_files:
        logging.warning("No .txt pack-list files found in %s", RESOURCE_PACKS_DIR)
        return []

    created: list[Path] = []
    for txt in txt_files:
        name = txt.stem                         # e.g. "calibrations"
        zip_path = out / f"{name}.zip"

        if zip_path.exists() and not overwrite:
            logging.info("Already exists, skipping: %s", zip_path)
            created.append(zip_path)
            continue

        entries = _parse_pack_list(txt)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in entries:
                src = CACHE_PATH / rel
                if src.is_file():
                    zf.write(src, arcname=rel)
                else:
                    logging.warning("Listed but missing, skipped: %s", src)

            # Include the pack-list inside the archive
            zf.write(txt, arcname=f"resource_packs/{txt.name}")

        logging.info("Packed %s (%d entries, %d bytes)",
                     zip_path, len(entries), zip_path.stat().st_size)
        created.append(zip_path)

    return created


def build_registry(scan_dir: Path | None = None) -> Path:
    """
    Scan *scan_dir* (default: TEMP_DIR) for ``.zip`` archives produced by
    ``pack_all`` and write a ``registry.json`` next to them.

    Each entry contains:
      - ``sha256``  - hex digest of the archive
      - ``file``    - archive filename
      - ``url``     - empty string placeholder (developer fills in GDrive link)

    Returns the path to the written registry file.
    """
    scan = scan_dir or TEMP_DIR
    reg: dict = {"version": 1, "packs": {}}

    for zp in sorted(scan.glob("*.zip")):
        reg["packs"][zp.stem] = {
            "file":   zp.name,
            "sha256": f"sha256:{_sha256(zp)}",
            "url":    "",  # <-- developer fills in GDrive file ID after upload
        }

    out = scan / "registry.json"
    out.write_text(json.dumps(reg, indent=2), "utf-8")
    logging.info("Registry written to %s (%d pack(s))", out, len(reg["packs"]))
    return out


# ══════════════════════════════════════════════════════════════
#  END-USER TOOLS - fetch & install
# ══════════════════════════════════════════════════════════════

def fetch_registry() -> dict:
    """
    Download the registry JSON from GDrive using the hard-coded
    ``REGISTRY_GDRIVE_ID``.  Returns the parsed dict.
    """
    if not REGISTRY_URL:
        raise RuntimeError("'registry_url' is not set in project_config.json. "
                           "A developer must upload the registry and set the URL.")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    local = Path(pooch.retrieve(
        url=_gdrive_download_url(REGISTRY_URL),
        known_hash=None,               # registry itself is not hash-checked
        fname="registry.json",
        path=str(TEMP_DIR),
        progressbar=False,
    ))
    return json.loads(local.read_text("utf-8"))


def fetch_resource_pack(name: str, registry: dict, overwrite: bool = False) -> None:
    """
    Download, verify, unpack, and distribute a single resource pack.

    Steps:
      1. Look up *name* in *registry* for the download URL and SHA-256.
      2. Download the ZIP via ``pooch.retrieve`` (hash-verified) into TEMP_DIR.
      3. Extract into a temp subfolder.
      4. Move every file to ``CACHE_PATH / <relative_path>`` as recorded in
         the archive; the ``.txt`` pack-list ends up in ``resource_packs/``.
      5. Create destination folders recursively when they don't exist.
      6. Remove the archive and temp extraction folder.
    """
    packs = registry.get("packs", {})
    if name not in packs:
        raise KeyError(f"'{name}' not in registry. Available: {list(packs)}")

    entry = packs[name]
    url = entry.get("url", "")
    if not url:
        raise ValueError(f"No download URL for '{name}' in registry.")

    # If the pack-list already exists locally, assume it's installed
    installed = RESOURCE_PACKS_DIR / f"{name}.txt"
    if installed.exists() and not overwrite:
        logging.info("'%s' already installed (found %s). Skipping.", name, installed)
        return

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_PACKS_DIR.mkdir(parents=True, exist_ok=True)

    # Download with SHA-256 verification
    zip_file = Path(pooch.retrieve(
        url=_gdrive_download_url(url),
        known_hash=entry["sha256"],
        fname=entry["file"],
        path=str(TEMP_DIR),
        progressbar=True,
    ))

    # Extract into a temp subfolder
    extract_dir = TEMP_DIR / name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    _safe_extract(zip_file, extract_dir)

    # Distribute every extracted file to its proper location under CACHE_PATH
    for item in extract_dir.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(extract_dir)
        dest = CACHE_PATH / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(dest))

    # Clean up temp artefacts
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_file.unlink(missing_ok=True)
    logging.info("Installed resource pack '%s'.", name)


def fetch_all_resource_packs(overwrite: bool = False) -> None:
    """Download and install every pack listed in the remote registry."""
    registry = fetch_registry()
    for name in registry.get("packs", {}):
        fetch_resource_pack(name, registry, overwrite=overwrite)


def sync_resource_packs() -> None:
    """
    Ensure local resource packs are in sync with the remote registry.

    - On first run: downloads everything.
    - On subsequent runs: compares remote SHA-256 hashes against the local
      registry cache; only re-downloads packs whose hashes changed.
    - If the remote registry is unreachable or ``registry_url`` is not
      configured, emits a warning and returns silently.
    """
    if not REGISTRY_URL:
        return

    try:
        remote_reg = fetch_registry()
    except Exception:
        import warnings
        warnings.warn("Could not fetch remote resource registry. Using cached data.")
        return

    # Load previously cached registry (if any) to detect changes
    local_reg_path = RESOURCE_PACKS_DIR / "registry.json"
    local_packs: dict = {}
    if local_reg_path.exists():
        try:
            local_packs = json.loads(local_reg_path.read_text("utf-8")).get("packs", {})
        except Exception:
            pass

    remote_packs = remote_reg.get("packs", {})

    for name, entry in remote_packs.items():
        local_entry = local_packs.get(name, {})
        # Skip if hash matches (already up-to-date)
        if local_entry.get("sha256") == entry.get("sha256"):
            installed = RESOURCE_PACKS_DIR / f"{name}.txt"
            if installed.exists():
                continue

        try:
            fetch_resource_pack(name, remote_reg, overwrite=True)
        except Exception:
            pass  # silently skip failed packs

    # Cache the remote registry locally for future comparisons
    RESOURCE_PACKS_DIR.mkdir(parents=True, exist_ok=True)
    local_reg_path.write_text(json.dumps(remote_reg, indent=2), "utf-8")


#%%
# # ──────────────────────────────────────────────────────────────
# #  Developer helper: pack resources and build registry
# #  Uncomment and run this cell to regenerate ZIPs + registry.
# # ──────────────────────────────────────────────────────────────

# logging.basicConfig(level=logging.INFO)

# # 1. Create ZIP archives from every .txt pack-list in resource_packs/
# zips = pack_all(overwrite=True)

# # 2. Build registry.json next to the ZIPs (in temp/)
# reg_path = build_registry()

# # 3. Upload the ZIPs and registry.json to GDrive manually, then:
# #    a) Open the generated registry.json
# #    b) Fill in the "url" fields with GDrive file IDs
# #    c) Re-upload the updated registry.json
# #    d) Paste its GDrive file ID into REGISTRY_GDRIVE_ID at the top of this file


