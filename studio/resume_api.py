import hmac
import os

from fastapi import APIRouter, Header, HTTPException

from .train_engine import resume_training

router = APIRouter()


@router.post("/api/studio/jobs/{job_id}/resume")
def resume_job(
    job_id: str,
    authorization: str = Header(default="")
):
    token = os.environ.get("DGPU_TOKEN", "")

    if not token or not hmac.compare_digest(
        authorization, "Bearer " + token
    ):
        raise HTTPException(401, "Invalid token")

    try:
        return resume_training(job_id)
    except ValueError as error:
        raise HTTPException(409, str(error))
