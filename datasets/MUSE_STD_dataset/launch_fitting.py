"""
Launches multiple STD_fitter.py jobs in parallel, distributing samples evenly across CUDA devices.

Usage:
    python launch_fitting.py --n_jobs 4 --devices cuda:0 cuda:1
    python launch_fitting.py --n_jobs 6 --devices cuda:0 cuda:1 cuda:2 --output_folder F:/ESO/Data/MUSE

Jobs are round-robin assigned to the specified devices. Each job gets a contiguous
slice of sample IDs. All jobs run concurrently as subprocesses.
"""

import sys
import argparse
import pickle
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def get_good_ids():
    """Load the list of valid sample IDs from the dataset."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from STD_dataset_utils import STD_FOLDER

    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
        psf_df = pickle.load(f)

    psf_df = psf_df[psf_df['Corrupted']   == False]
    psf_df = psf_df[psf_df['Bad quality'] == False]
    return psf_df.index.values.tolist()


def main():
    parser = argparse.ArgumentParser(description='Launch parallel STD_fitter.py jobs across CUDA devices.')
    parser.add_argument('--n_jobs',  type=int, required=True, help='Number of parallel fitting jobs')
    parser.add_argument('--devices', type=str, nargs='+', required=True, help='CUDA devices, e.g. cuda:0 cuda:1')
    parser.add_argument('--output_folder', type=str, default=None, help='Optional output folder for fitted parameters')
    args = parser.parse_args()

    good_ids = get_good_ids()
    n_samples = len(good_ids)
    n_jobs = min(args.n_jobs, n_samples)

    # Split IDs into n_jobs contiguous chunks
    chunk_size = n_samples // n_jobs
    remainder  = n_samples % n_jobs

    chunks = []
    offset = 0
    for i in range(n_jobs):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(good_ids[offset:offset + size])
        offset += size

    # Launch subprocesses
    fitter_script = str(SCRIPT_DIR / 'STD_fitter.py')
    processes = []

    print(f"Launching {n_jobs} jobs across devices {args.devices} for {n_samples} samples\n")

    for i, chunk in enumerate(chunks):
        device = args.devices[i % len(args.devices)]
        start_id = chunk[0]
        end_id   = chunk[-1]

        cmd = [sys.executable, fitter_script, device, str(start_id), str(end_id)]
        if args.output_folder:
            cmd.append(args.output_folder)

        print(f"  Job {i}: device={device}, IDs {start_id}..{end_id} ({len(chunk)} samples)")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        processes.append((i, proc))

    print(f"\nAll {n_jobs} jobs started. Waiting for completion...")

    # Wait for all jobs and report exit codes
    failures = []
    for i, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            failures.append((i, proc.returncode))

    if failures:
        print(f"\n{len(failures)} job(s) failed:")
        for i, rc in failures:
            print(f"  Job {i}: exit code {rc}")
        sys.exit(1)
    else:
        print("\nAll jobs completed successfully.")


if __name__ == '__main__':
    main()
