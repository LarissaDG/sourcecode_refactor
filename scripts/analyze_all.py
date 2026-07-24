"""
Roda as duas versões de análise em sequência:
  1. Fiel ao ICCC (metricas.py) → figures_iccc/ + iccc_stats_report.txt
  2. Nova versão (Friedman + CLD) → figures/ + samples/

Uso:
    python scripts/analyze_all.py --config configs/analysis_local.yaml
    python scripts/analyze_all.py --config configs/analysis.yaml --skip-samples
"""

import argparse
import subprocess
import sys
import os


def run(script, config, extra_args):
    cmd = [sys.executable, script, "--config", config] + extra_args
    print(f"\n{'='*60}")
    print(f"Rodando: {' '.join(cmd)}")
    print("="*60)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print(f"[AVISO] {script} terminou com código {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--skip-samples", action="store_true",
                        help="Pula amostras visuais na análise nova")
    parser.add_argument("--skip-iccc", action="store_true",
                        help="Pula a versão ICCC fiel")
    parser.add_argument("--skip-new", action="store_true",
                        help="Pula a versão nova (Friedman + CLD)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_iccc:
        run(os.path.join(base, "analyze_iccc.py"), args.config, [])

    if not args.skip_new:
        extra = ["--skip-samples"] if args.skip_samples else []
        run(os.path.join(base, "analyze.py"), args.config, extra)

    print("\n✓ Análise completa.")
    print("  figures_iccc/ → metodologia ICCC (t-test, Mann-Whitney, ANOVA, radar)")
    print("  figures/      → nova metodologia (Friedman + Wilcoxon + CLD)")


if __name__ == "__main__":
    main()
