"""
Summary figure of the TIPTOP vs TipTorch performance tests from their JSON records in tests/runs/performance/ (backend, polychromatic
and batched tests): one panel per test with the median run times (log scale) and the speed-ups, plus a combined markdown table, for
presentations. Run after the tests: python tests/plot_tiptop_performance.py  ->  tests/runs/performance/summary.{png,svg,md}
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiptop_performance_utils import INK, INK_SOFT, OUTPUT, draw_bars, figure_height, markdown_table

TESTS = ('backend_performance', 'polychromatic_performance', 'batched_performance')


def main():
    records = {t: json.load(open(OUTPUT / f'{t}.json')) for t in TESTS if (OUTPUT / f'{t}.json').exists()}
    assert records, f'no results in {OUTPUT}: run the tests first'

    heights = [figure_height(r['rows']) for r in records.values()]
    fig, axes = plt.subplots(len(records), 1, figsize=(12, sum(heights) + 0.3), gridspec_kw=dict(height_ratios=heights))
    axes = [axes] if len(records) == 1 else list(axes)
    table = []
    for ax, rec in zip(axes, records.values()):
        draw_bars(ax, rec['rows'], rec.get('gains'))
        ax.set_title(rec['title'], loc='left', fontsize=10.5, color=INK)
        table.append(markdown_table(rec['title'] + (f" ({rec['subtitle']})" if rec.get('subtitle') else ''), rec['rows'], rec.get('gains')))
    axes[-1].set_xlabel('run time [s], median over repeats (log scale)', color=INK_SOFT, fontsize=9.5)
    fig.suptitle(f"TIPTOP vs TipTorch run times ({next(iter(records.values()))['device']})", x=0.02, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT / 'summary.png', dpi=160)
    fig.savefig(OUTPUT / 'summary.svg')
    (OUTPUT / 'summary.md').write_text('\n'.join(table))
    print(f'summary written to {OUTPUT / "summary"}.png / .svg / .md')


if __name__ == '__main__':
    main()
