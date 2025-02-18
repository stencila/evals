# Ignore Meta overrides for Tortoise models
# pyright: reportIncompatibleVariableOverride=false

from enum import StrEnum
from pathlib import Path
from decimal import Decimal

from pydantic import BaseModel
from tortoise import Model, Tortoise, fields

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_CSV = BASE_DIR / "tables" / "model.csv"
PROMPT_CSV = BASE_DIR / "tables" / "prompt.csv"


class BenchmarkType(StrEnum):
    """Kind of score."""

    COST = "cost"
    SPEED = "speed"
    QUALITY = "quality"


class Benchmark(BaseModel):
    """Model score."""

    # Model name. Must match the name in the model table.
    model: str

    # This could be:
    # - a category if it is quality score.
    # - a host if it is cost or speed score.
    # - "*" is a catchall that indicates everything (all hosts or any category).
    context: str = "*"

    # The score!
    score: float | Decimal


class BenchmarkResult(BaseModel):
    """Benchmark result."""

    bm_type: BenchmarkType
    unit: str
    scores: list[Benchmark]


async def init_connection(db_url: str):
    await Tortoise.init(db_url=db_url, modules={"models": ["evals.orm"]})
    await Tortoise.generate_schemas()


async def close_connection():
    await Tortoise.close_connections()


class Database:
    """Helper class to manage connection to the database."""

    def __init__(self, path: Path):
        self.url = f"sqlite://{path}"

    async def __aenter__(self):
        await init_connection(self.url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb  # prevent unused warnings.
        await close_connection()


class PromptRecord(Model):
    id = fields.TextField(pk=True)
    category = fields.TextField()

    # Reverse relations
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "prompt"


class ModelRecord(Model):
    id = fields.TextField(pk=True)

    # Reverse relations
    model_scores: fields.ReverseRelation["ModelScoreRecord"]
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "model"


class ModelScoreRecord(Model):
    id = fields.IntField(pk=True)
    model = fields.ForeignKeyField("models.ModelRecord", related_name="model_scores")
    cost_score = fields.FloatField()
    speed_score = fields.FloatField()

    class Meta:
        table = "model_score"


class ModelPromptScore(Model):
    id = fields.IntField(pk=True)
    model = fields.ForeignKeyField(
        "models.ModelRecord", related_name="model_prompt_scores"
    )
    prompt = fields.ForeignKeyField(
        "models.PromptRecord", related_name="model_prompt_scores"
    )
    quality_score = fields.SmallIntField()

    class Meta:
        table = "model_prompt_scores"
