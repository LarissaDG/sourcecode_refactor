import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.noise import NOISE_FNS, DEFAULT_NOISE_LEVELS, DEFAULT_NOISE_TYPES


# Os 10 atributos estéticos do APDDv2 (sem o score total)
AESTHETIC_ATTRIBUTES = [
    "Theme and logic",
    "Creativity",
    "Layout and composition",
    "Space and perspective",
    "The sense of order",
    "Light and shadow",
    "Color",
    "Details and texture",
    "The overall",
    "Mood",
]

# Os 9 atributos usados para calcular o score médio na amostragem por bins
# (exclui "The sense of order" — modelo 6 do ArtCLIP tem bug conhecido e é
# excluído da avaliação, ver README)
BIN_ATTRIBUTES = [a for a in AESTHETIC_ATTRIBUTES if a != "The sense of order"]

# Possíveis nomes de colunas no CSV (CSV oficial vs. fixtures de teste)
SCORE_COL_CANDIDATES = ["Total aesthetic score", "Score"]
CATEGORY_COL_CANDIDATES = ["Artistic Categories", "category"]
COMMENT_COL_CANDIDATES = ["Language Comment", "comment"]


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _largest_remainder_allocation(bin_counts: pd.Series, n: int) -> pd.Series:
    """
    Aloca `n` unidades entre bins proporcionalmente ao tamanho de cada bin
    (amostragem estratificada proporcional), usando o método dos maiores
    restos para fechar exatamente em `n` sem exceder o tamanho de nenhum bin.
    """
    proportions = bin_counts / bin_counts.sum()
    raw = proportions * n
    alloc = np.minimum(np.floor(raw).astype(int), bin_counts)

    remainder = n - int(alloc.sum())
    fracs = (raw - alloc).sort_values(ascending=False)
    for b in fracs.index:
        if remainder <= 0:
            break
        if alloc[b] < bin_counts[b]:
            alloc[b] += 1
            remainder -= 1

    # Se algum resto ainda sobrar (bins pequenos já saturados), distribui
    # nos bins com folga restante, na ordem em que aparecem.
    if remainder > 0:
        for b in bin_counts.index:
            if remainder <= 0:
                break
            slack = int(bin_counts[b] - alloc[b])
            take = min(slack, remainder)
            alloc[b] += take
            remainder -= take

    return alloc


def _format_legacy_bin_report(sampled_df: pd.DataFrame, bin_cols: list, legacy_csv: str,
                               n_found: int, n_total: int, n_bins: int) -> str:
    lines = [
        "Distribuicao da amostra legada do ICCC (proportional_stratified via legacy_csv)",
        f"Fonte: {legacy_csv}",
        f"Atributos usados no binning: {', '.join(bin_cols)}",
        f"Imagens do CSV legado encontradas no dataset atual: {n_found}/{n_total}",
        "",
    ]
    if not bin_cols or sampled_df.empty:
        lines.append("(sem atributos estéticos disponíveis para o binning)")
        return "\n".join(lines) + "\n"

    mean_score = sampled_df[bin_cols].mean(axis=1)
    n_bins_eff = max(1, min(n_bins, mean_score.nunique(), len(sampled_df)))
    cat = pd.cut(mean_score, bins=n_bins_eff, duplicates="drop")
    bin_counts = cat.value_counts().sort_index()

    lines.append(f"{'Faixa de score':^20} | {'Total':>7}")
    lines.append("-" * 32)
    for edge, count in bin_counts.items():
        lines.append(f"[{edge.left:.2f}, {edge.right:.2f}]".center(20) + f" | {count:>7}")
    return "\n".join(lines) + "\n"


def _format_bin_report(bin_counts: pd.Series, alloc: pd.Series, bin_cols: list,
                        bin_edges) -> str:
    lines = [
        "Distribuicao da amostragem proporcional estratificada (proportional_stratified)",
        f"Atributos usados no binning: {', '.join(bin_cols)}",
        f"Numero de bins: {len(bin_edges)}",
        f"Total de imagens no dataset: {int(bin_counts.sum())}",
        f"Total amostrado: {int(alloc.sum())}",
        "",
        f"{'Bin':>4} | {'Faixa de score':^20} | {'Total':>7} | {'Amostrado':>10} | {'Proporcao':>10}",
        "-" * 66,
    ]
    for b in bin_counts.index:
        edge = bin_edges[b] if 0 <= b < len(bin_edges) else None
        faixa = f"[{edge.left:.2f}, {edge.right:.2f}]" if edge is not None else "?"
        total = int(bin_counts[b])
        sampled = int(alloc[b])
        prop = sampled / total if total else 0.0
        lines.append(f"{b:>4} | {faixa:^20} | {total:>7} | {sampled:>10} | {prop:>9.1%}")
    return "\n".join(lines) + "\n"


class APDDv2Dataset(Dataset):
    """
    Loader para o APDDv2.

    Estrutura esperada no disco:
        <root>/
            APDDv2-10023.csv
            images/
                painting_001.jpg
                painting_002.jpg
                ...

    O CSV contém: filename, score total, atributos estéticos e comentários linguísticos.
    """

    def __init__(self, root: str, split: str = "all", transform=None):
        """
        Args:
            root:      Caminho raiz do dataset (onde está o CSV e a pasta images/).
            split:     "all" | "train" | "test" — usa a coluna 'split' do CSV se existir.
            transform: Transformações torchvision. Se None, usa o padrão para CLIP.
        """
        self.root = root
        # Aceita variações do nome/estrutura da pasta de imagens
        for _candidate in ("images", "APDDv2images", os.path.join("APDDv2images", "APDDv2images")):
            _candidate_path = os.path.join(root, _candidate)
            if os.path.isdir(_candidate_path):
                self.images_dir = _candidate_path
                break
        else:
            self.images_dir = os.path.join(root, "images")

        csv_path = os.path.join(root, "APDDv2-10023.csv")
        self.df = pd.read_csv(csv_path, encoding="ISO-8859-1")

        self.score_col = _first_present(self.df.columns, SCORE_COL_CANDIDATES)
        self.category_col = _first_present(self.df.columns, CATEGORY_COL_CANDIDATES)
        self.comment_col = _first_present(self.df.columns, COMMENT_COL_CANDIDATES)
        self.bin_cols = [c for c in BIN_ATTRIBUTES if c in self.df.columns]

        if self.score_col:
            self.df = self.df.dropna(subset=[self.score_col])

        if split != "all" and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Filtra linhas cujo arquivo não existe no disco (dataset parcialmente disponível)
        mask = self.df["filename"].apply(
            lambda f: os.path.exists(self._resolve_path(str(f).strip()))
        )
        n_total = len(self.df)
        self.df = self.df[mask].reset_index(drop=True)
        if len(self.df) < n_total:
            import warnings
            warnings.warn(
                f"APDDv2Dataset: {n_total - len(self.df)} imagens ausentes no disco "
                f"(de {n_total}). Usando {len(self.df)} disponíveis.",
                RuntimeWarning,
            )

        self.transform = transform or self._default_transform()
        self.bin_report = None

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = str(row["filename"]).strip()
        path = self._resolve_path(filename)
        image = Image.open(path).convert("RGB")

        noise_type  = row.get("noise_type")  if hasattr(row, "get") else None
        noise_level = int(row["noise_level"]) if "noise_level" in row.index and pd.notna(row["noise_level"]) else 0

        if noise_type and noise_level > 0:
            np.random.seed(noise_level + idx)
            image = NOISE_FNS[noise_type](image, noise_level)

        image_t = self.transform(image)

        sample = {
            "image":       image_t,
            "filename":    filename,
            "path":        path,
            "noise_type":  noise_type or "none",
            "noise_level": noise_level,
        }

        sample["score"] = float(row[self.score_col]) if self.score_col else float("nan")

        # Sempre inclui todas as chaves para manter dicts do batch consistentes
        for attr in AESTHETIC_ATTRIBUTES:
            val = row[attr] if attr in row.index else float("nan")
            sample[attr.lower()] = float(val) if pd.notna(val) else float("nan")

        sample["caption"] = (
            str(row[self.comment_col])
            if self.comment_col and pd.notna(row[self.comment_col])
            else ""
        )
        sample["category"] = (
            str(row[self.category_col])
            if self.category_col and pd.notna(row[self.category_col])
            else ""
        )

        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> str:
        path = os.path.join(self.images_dir, filename)
        if not os.path.exists(path):
            # Tenta extensão alternativa
            base = os.path.splitext(filename)[0]
            for ext in (".png", ".jpg", ".jpeg"):
                alt = os.path.join(self.images_dir, base + ext)
                if os.path.exists(alt):
                    path = alt
                    break
        return path

    def _load_legacy_csv_sample(self, legacy_csv: str) -> pd.DataFrame:
        """
        Carrega uma amostra FIXA a partir de um CSV externo (ex: sampled_dataset.csv
        do ICCC original), em vez de recalcular a amostragem — usado para reproduzir
        exatamente a seleção de imagens de um experimento legado. Casa por stem do
        filename com o dataset atual; itens do CSV sem correspondência são ignorados
        (com aviso).
        """
        legacy_df = pd.read_csv(legacy_csv, encoding="ISO-8859-1")
        fn_col = _first_present(legacy_df.columns, ["filename", "Filename", "Numero da Obra", "image"])
        if not fn_col:
            raise ValueError(
                f"Coluna de filename não encontrada em {legacy_csv}. "
                f"Colunas disponíveis: {list(legacy_df.columns)}"
            )

        legacy_stems = {
            os.path.splitext(str(v).strip())[0]
            for v in legacy_df[fn_col]
            if pd.notna(v)
        }

        current_stems = self.df["filename"].apply(lambda f: os.path.splitext(str(f).strip())[0])
        sampled_df = self.df[current_stems.isin(legacy_stems)]

        n_found, n_total = len(sampled_df), len(legacy_stems)
        if n_found < n_total:
            import warnings
            warnings.warn(
                f"_load_legacy_csv_sample: {n_total - n_found} imagens do CSV legado "
                f"({legacy_csv}) não encontradas no dataset atual. Usando {n_found}/{n_total}.",
                RuntimeWarning,
            )

        return sampled_df, n_found, n_total

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """Normalização padrão do CLIP."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std= [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _make_subset(self, sampled_df: pd.DataFrame, bin_report: str = None) -> "APDDv2Dataset":
        subset = APDDv2Dataset.__new__(APDDv2Dataset)
        subset.root        = self.root
        subset.images_dir  = self.images_dir
        subset.df          = sampled_df.reset_index(drop=True)
        subset.transform   = self.transform
        subset.score_col   = self.score_col
        subset.category_col = self.category_col
        subset.comment_col = self.comment_col
        subset.bin_cols    = self.bin_cols
        subset.bin_report  = bin_report
        return subset

    # ------------------------------------------------------------------
    # Amostragem — chamada pela Caixinha 1
    # ------------------------------------------------------------------

    def sample(self, n: int, strategy: str = "random", seed: int = 42, n_bins: int = 30,
               noise_levels=None, noise_types=None, legacy_csv: str = None,
               **kwargs) -> "APDDv2Dataset":
        """
        Retorna um subconjunto do dataset.

        Args:
            n:        Número de amostras desejadas.
            strategy: "random"                 — amostragem aleatória simples.
                      "stratified"              — balanceia por categoria artística.
                      "uniform_bins"            — calcula a média aritmética dos 9
                                                    atributos estéticos (BIN_ATTRIBUTES),
                                                    divide o range em `n_bins` faixas de
                                                    igual largura e amostra uniformemente
                                                    entre elas (mesma contagem por bin).
                      "proportional_stratified" — mesmo binning acima, mas aloca a
                                                    amostra proporcionalmente ao tamanho
                                                    de cada bin (preserva a distribuição
                                                    original do dataset). O subset
                                                    retornado carrega `.bin_report`, um
                                                    texto com a distribuição por bin.
                                                    Se `legacy_csv` for informado, NÃO
                                                    recalcula — carrega a amostra fixa
                                                    daquele CSV (ex: reproduzir o
                                                    experimento ICCC original).
            seed:        Semente para reprodutibilidade.
            n_bins:      Número de faixas usadas pelas estratégias baseadas em bins.
            legacy_csv:  Caminho de um CSV externo com a coluna de filename da amostra
                         a reusar (só tem efeito em strategy="proportional_stratified").
        """
        bin_report = None
        if strategy == "random":
            sampled_df = self.df.sample(n=n, random_state=seed)

        elif strategy == "stratified":
            if not self.category_col:
                raise ValueError("Coluna de categoria não encontrada para amostragem estratificada.")
            n_categories = self.df[self.category_col].nunique()
            per_category = max(1, n // n_categories)
            sampled_df = (
                self.df
                .groupby(self.category_col, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_category), random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )

        elif strategy == "uniform_bins":
            if not self.bin_cols:
                raise ValueError("Nenhum dos atributos estéticos usados para o binning foi encontrado no CSV.")

            mean_score = self.df[self.bin_cols].mean(axis=1)
            n_bins_eff = max(1, min(n_bins, mean_score.nunique(), len(self.df)))

            df_binned = self.df.copy()
            df_binned["_bin"] = pd.cut(mean_score, bins=n_bins_eff, labels=False, duplicates="drop")

            per_bin = max(1, n // df_binned["_bin"].nunique())
            sampled_df = (
                df_binned
                .groupby("_bin", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle final
                .head(n)
            )
            sampled_df = self.df.loc[sampled_df.index]

        elif strategy == "proportional_stratified":
            if legacy_csv:
                sampled_df, n_found, n_total = self._load_legacy_csv_sample(legacy_csv)
                bin_report = _format_legacy_bin_report(
                    sampled_df, self.bin_cols, legacy_csv, n_found, n_total, n_bins
                )
            else:
                if not self.bin_cols:
                    raise ValueError("Nenhum dos atributos estéticos usados para o binning foi encontrado no CSV.")

                mean_score = self.df[self.bin_cols].mean(axis=1)
                n_bins_eff = max(1, min(n_bins, mean_score.nunique(), len(self.df)))

                cat = pd.cut(mean_score, bins=n_bins_eff, duplicates="drop")
                df_binned = self.df.copy()
                df_binned["_bin"] = cat.cat.codes
                bin_edges = cat.cat.categories

                bin_counts = df_binned.loc[df_binned["_bin"] >= 0, "_bin"].value_counts().sort_index()
                n_eff = min(n, int(bin_counts.sum()))
                alloc = _largest_remainder_allocation(bin_counts, n_eff)

                parts = [
                    df_binned[df_binned["_bin"] == b].sample(n=int(cnt), random_state=seed)
                    for b, cnt in alloc.items() if cnt > 0
                ]
                sampled_df = pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle final
                sampled_df = self.df.loc[sampled_df.index]
                bin_report = _format_bin_report(bin_counts, alloc, self.bin_cols, bin_edges)

        else:
            raise ValueError(
                f"Estratégia desconhecida: '{strategy}'. "
                "Use 'random', 'stratified', 'uniform_bins' ou 'proportional_stratified'."
            )

        # Expansão por ruído: cada imagem × tipo × nível
        if noise_levels is not None or noise_types is not None:
            levels = list(noise_levels) if noise_levels is not None else DEFAULT_NOISE_LEVELS
            types  = list(noise_types)  if noise_types  is not None else DEFAULT_NOISE_TYPES
            rows = []
            for _, row in sampled_df.iterrows():
                for noise_type in types:
                    for level in levels:
                        r = row.to_dict()
                        r["noise_type"]  = noise_type
                        r["noise_level"] = level
                        rows.append(r)
            sampled_df = pd.DataFrame(rows)

        return self._make_subset(sampled_df, bin_report=bin_report)
