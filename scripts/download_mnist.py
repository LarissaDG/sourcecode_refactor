"""
Baixa o MNIST, embaralha e amostra 500 dígitos aleatoriamente.

Uso:
    python3 scripts/download_mnist.py --out data/mnist
    python3 scripts/download_mnist.py --out data/mnist --n 100 --seed 0

Flags:
    --n      Número de amostras (padrão: 500)
    --seed   Semente aleatória (padrão: 42)
    --split  train | test | all (padrão: all)

Dependências:
    pip install torchvision pillow pandas
"""

import argparse
from pathlib import Path

import pandas as pd
from torchvision.datasets import MNIST


def main():
    parser = argparse.ArgumentParser(description="Download e amostragem do MNIST")
    parser.add_argument("--out",   required=True,        help="Diretório de saída")
    parser.add_argument("--n",     type=int, default=500, help="Número de amostras (padrão: 500)")
    parser.add_argument("--seed",  type=int, default=42,  help="Semente aleatória (padrão: 42)")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "Imagens"
    images_dir.mkdir(exist_ok=True)

    # Baixa o MNIST via torchvision (raw, sem transform)
    print("Baixando MNIST...")
    if args.split == "all":
        splits = [
            MNIST(root=str(out / "_raw"), train=True,  download=True),
            MNIST(root=str(out / "_raw"), train=False, download=True),
        ]
        data   = [(img, label) for ds in splits for img, label in ds]
    elif args.split == "train":
        ds   = MNIST(root=str(out / "_raw"), train=True,  download=True)
        data = list(ds)
    else:
        ds   = MNIST(root=str(out / "_raw"), train=False, download=True)
        data = list(ds)

    print(f"  {len(data)} imagens disponíveis.")

    # Embaralha e amostra
    import random
    random.seed(args.seed)
    random.shuffle(data)
    amostra = data[:args.n]
    print(f"  Amostrando {len(amostra)} imagens (seed={args.seed})...")

    # Salva imagens e CSV
    registros = []
    for i, (img, label) in enumerate(amostra):
        filename = f"{i:04d}_label{label}.png"
        img.save(images_dir / filename)
        registros.append({"filename": filename, "label": int(label), "index": i})

    df = pd.DataFrame(registros)
    csv_path = out / "mnist_sample.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nFINALIZADO")
    print(f"  {len(amostra)} imagens em {images_dir}/")
    print(f"  Metadados em {csv_path}")
    print(f"  Distribuição de labels:\n{df['label'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
