"""Build the scores.

For each model and category, we have a quality, cost, and speed score.
Together these can be used to generate the ideal models, given a weighting.

"""

# TODO: for now, we ignore *hosts*, that a model can be hosted by one or more
# platforms. We can add this later.

import decimal
import os
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, RootModel

from .benchmarks import all_benchmarks
from .modelmap import ModelMapper
from .orm import Benchmark, BenchmarkResult, BenchmarkType
from .settings import get_settings


def get_scores_like(
    results: list[BenchmarkResult], bm_type: BenchmarkType
) -> tuple[str, list[Benchmark]]:
    chosen = [r for r in results if r.bm_type == bm_type]
    # There should only be one quality score.
    match len(chosen):
        case 1:
            return chosen[0].unit, chosen[0].scores

        case 0:
            msg = "No quality score found."
            raise ValueError(msg)

        case _:
            msg = "Multiple quality scores found."
            raise ValueError(msg)


class ScoreRecord(BaseModel):
    model: str

    # TODO: We should use an enum here later.
    category: str

    quality: float | None
    quality_rank: int | None = None
    quality_pct: float | None = None
    quality_unit: str

    cost: decimal.Decimal | None
    cost_rank: int | None = None
    cost_pct: float | None = None
    cost_unit: str | None

    speed: float | None
    speed_rank: int | None = None
    speed_pct: float | None = None
    speed_unit: str | None


class ScoreRecordList(RootModel):
    root: list[ScoreRecord]


def calculate_geometric_means(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    cols = list(cols)

    def geom_mean(row: tuple[float]):
        row = np.array(row, dtype=np.float64)  # type: ignore
        return np.exp(np.mean(np.log(row)))

    df_overall = df.with_columns(
        [
            pl.struct(cols)
            .map_elements(
                lambda x: geom_mean(tuple(x.values())), return_dtype=pl.Float64
            )
            .alias("geometric_avg_rank")
        ]
    )

    overall = df_overall.get_column("geometric_avg_rank")
    df = df.with_columns([overall])

    return df


def rank_by_category(df: pl.DataFrame) -> pl.DataFrame:
    quality_ranked = (
        pl.col("quality")
        .rank(method="min", descending=True)
        # .over("category") # TODO: quality context
        .alias("quality_rank")
    )

    speed_ranked = (
        pl.col("speed")
        .rank(method="min", descending=True)
        # .over("category") # TODO: speed context
        .alias("speed_rank")
    )

    cost_ranked = (
        pl.col("cost")
        .rank(method="min", descending=False)
        # .over("category") # TODO: cost context
        .alias("cost_rank")
    )
    df = df.with_columns([quality_ranked, speed_ranked, cost_ranked])

    return df

def calculate_percentiles(scores: List[float], invert_scores = False) -> np.ndarray:
    obs = np.array(scores)

    # lower is better for cost
    # TODO: check whether this is already done earlier
    if invert_scores:
        obs = decimal.Decimal("1.0") - obs

    sorted_obs = np.sort(obs)
    positions = np.searchsorted(sorted_obs, obs)
    percentiles = positions / len(obs)
    percentiles = percentiles * 100.0
    return percentiles

def generate_scores() -> ScoreRecordList:
    def report_missing(quality, cost, speed):
        missing = []
        if quality is None:
            missing.append("quality")
        if cost is None:
            missing.append("cost")
        if speed is None:
            missing.append("speed")

        if len(missing) > 0:
            part_1 = f"{model} is missing metric"
            match len(missing):
                case 1: part_2 = f": {missing[0]}."
                case 2: part_2 = f"s: {missing[0]} and {missing[1]}."
                case _: part_2 = f"s: {(', ').join(missing[0:-1])} and {missing[-1]}."
            logger.info(part_1 + part_2)

    mm = ModelMapper.load()
    results = [bm.get_benchmarks(mm) for bm in all_benchmarks().values()]
    q_unit, quality_scores = get_scores_like(results, BenchmarkType.QUALITY)
    c_unit, cost_scores = get_scores_like(results, BenchmarkType.COST)
    s_unit, speed_scores = get_scores_like(results, BenchmarkType.SPEED)

    quality_contexts = set(["overall"])
    data_by_model = defaultdict(dict)

    for score in quality_scores:
        # We know that full == overall from https://github.com/lm-sys/FastChat/blob/03d376ccb5048bcbaa49b24064a3534514a42faf/fastchat/serve/monitor/monitor_md.py#L55
        if score.context == "full":
            data_by_model[score.model][f"quality"] = score.score
        else:
            # TODO: skip these at a different place
            continue

    for score in cost_scores:
        # TODO: cost context
        data_by_model[score.model]["cost"] = score.score

    for score in speed_scores:
        # TODO: speed context, e.g. hosting provider
        data_by_model[score.model]["speed"] = score.score

    metrics = ["quality", "speed", "cost"]
    for metric in metrics:
        invert_scores = metric == "cost"
        model_names = []
        scores = []

        for model_name, data in data_by_model.items():
            score = data.get(metric)
            if score is None:
                continue
            scores.append(score)
            model_names.append(model_name)

        percentiles = calculate_percentiles(scores, invert_scores)
        for (model, pct) in zip(model_names, percentiles, strict=True):
            data_by_model[model][f"{metric}_pct"] = pct

    score_records = []
    for model, data in data_by_model.items():
        for context in quality_contexts:
            quality = data.get(f"quality")
            cost = data.get("cost")
            speed = data.get("speed")

            missing = []
            if quality is None:
                missing.append("quality")
            if cost is None:
                missing.append("cost")
            if speed is None:
                missing.append("speed")

            if len(missing) > 0:
                part_1 = f"{model} is missing metric"
                match len(missing):
                    case 1: part_2 = f": {missing[0]}."
                    case 2: part_2 = f"s: {missing[0]} and {missing[1]}."
                    case _: part_2 = f"s: {(', ').join(missing[0:-1])} and {missing[-1]}."  # impossible?
                logger.info(part_1 + part_2)

            score_records.append(
                ScoreRecord(
                    category=context,
                    quality=quality,
                    quality_unit=q_unit,
                    quality_pct=data.get(f"quality_pct"),
                    cost=cost,
                    cost_unit=c_unit,
                    cost_pct=data.get("cost_pct"),
                    speed=speed,
                    speed_unit=s_unit,
                    speed_pct=data.get("speed_pct"),
                    model=model,
                )
            )

    return ScoreRecordList(root=score_records)

# utility functions


def today():
    return date.today().strftime("%Y-%m-%d")


# disk-related


def symlink(original: Path, prefix="scoring"):
    parent_dir = original.parent
    symlink_path = parent_dir / f"{prefix}{original.suffix}"

    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    os.symlink(original.name, symlink_path)


def main():
    """Write parquet and JSON files with scoring data."""
    st = get_settings()
    home = st.get_routing_dir()

    score_card = generate_scores()
    df = pl.DataFrame(score_card.root)
    df = rank_by_category(df)
    df = calculate_geometric_means(df, ["quality_rank", "cost_rank", "speed_rank"])

    df.write_csv(home / f"scoring-with-incomplete-{today()}.csv")
    df.write_json(home / f"scoring-with-incomplete-{today()}.json")
    df.write_parquet(home / f"scoring-with-incomplete-{today()}.parquet")

    # update "scoring.(csv|json|parquet)" to always point to the most recent version
    symlink(home / f"scoring-with-incomplete-{today()}.csv", prefix="scoring-with-incomplete")
    symlink(home / f"scoring-with-incomplete-{today()}.json", prefix="scoring-with-incomplete")
    symlink(home / f"scoring-with-incomplete-{today()}.parquet", prefix="scoring-with-incomplete")

    df = df.drop_nulls(subset=pl.selectors.by_name("quality", "speed", "cost"))

    df.write_csv(home / f"scoring-{today()}.csv")
    df.write_json(home / f"scoring-{today()}.json")
    df.write_parquet(home / f"scoring-{today()}.parquet")

    # update "scoring.(csv|json|parquet)" to always point to the most recent version
    symlink(home / f"scoring-{today()}.csv", prefix="scoring")
    symlink(home / f"scoring-{today()}.json", prefix="scoring")
    symlink(home / f"scoring-{today()}.parquet", prefix="scoring")

    logger.debug(df)


if __name__ == "__main__":
    main()
