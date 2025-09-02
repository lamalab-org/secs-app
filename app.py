import time
from typing import Any

from fastapi import FastAPI
from loguru import logger
from main import spec2struct
from pydantic import BaseModel

app = FastAPI()


class GenerateRequest(BaseModel):
    mf: str
    spectrum: dict[str, list[float]]
    model: str = "residual"
    configs_path: str = "configs"
    ga_ir_exp: str | None = None
    ga_cnmr_exp: str | None = None
    ga_hnmr_exp: str | None = "test/hnmr_augment_finetune_residual"
    ga_hsqc_exp: str | None = None
    seed: int = 42
    init_pop_ga: int = 512
    frac_graph_ga_mutate: float = 0.3
    gens_ga: int = 10
    offspring_ga: int = 1024
    pop_ga: int = 512
    initial_environment: dict[str, str] | None = None


@app.post("/")
def elucidate_spectrum(request: GenerateRequest) -> list[dict[str, Any]]:
    # Use .model_dump() to convert the Pydantic model to a dict for secs
    start_time = time.time()
    results = spec2struct(**request.model_dump())
    end_time = time.time()
    logger.info(f"Elucidation took {end_time - start_time:.2f} seconds")
    return results
