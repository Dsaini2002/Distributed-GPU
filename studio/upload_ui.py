import hmac
import os
import re
import uuid

from pathlib import Path

from fastapi import (
    APIRouter, UploadFile, File,
    Form, Header, HTTPException
)
from fastapi.responses import HTMLResponse

from .model_inspector import (
    inspect_model, inspect_dataset
)

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent

LIMITS = {
    "model": 2 * 1024 * 1024,
    "dataset": 20 * 1024 * 1024
}


@router.post("/api/studio/upload")
async def upload(
    kind: str = Form(...),
    file: UploadFile = File(...),
    authorization: str = Header(default="")
):
    token = os.environ.get("DGPU_TOKEN", "")

    if not token or not hmac.compare_digest(
        authorization, "Bearer " + token
    ):
        raise HTTPException(401, "Invalid token")

    if kind not in LIMITS:
        raise HTTPException(400, "Invalid upload type")

    original = (
        (file.filename or "")
        .replace("\\", "/")
        .split("/")[-1]
    )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",
        original
    ):
        raise HTTPException(400, "Invalid filename")

    extension = ".py" if kind == "model" else ".csv"

    if Path(original).suffix.lower() != extension:
        raise HTTPException(
            400, f"Expected {extension} file"
        )

    folder = ROOT / (
        "models" if kind == "model" else "data"
    )
    folder.mkdir(parents=True, exist_ok=True)

    output_name = (
        Path(original).stem
        + "_"
        + uuid.uuid4().hex[:8]
        + extension
    )

    destination = folder / output_name
    max_bytes = LIMITS[kind]
    size = 0

    try:
        with destination.open("xb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > max_bytes:
                    raise ValueError("Upload too large")

                output.write(chunk)

        if size == 0:
            raise ValueError("Empty file")

        if kind == "model":
            inspection = inspect_model(destination)
        else:
            inspection = inspect_dataset(destination)

    except Exception as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(422, str(error))

    finally:
        await file.close()

    return {
        "success": True,
        "kind": kind,
        "filename": output_name,
        "bytes": size,
        "inspection": inspection
    }


@router.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Distributed-GPU Upload</title>
<style>
body {
  background:#101827; color:#eff6ff;
  font-family:Arial,sans-serif; padding:35px;
}
main {max-width:760px;margin:auto}
h1 {color:#77baff}
.card {
  background:#1c2940; padding:22px;
  border-radius:12px; margin:15px 0;
}
label {display:block;margin-top:16px}
input,select {
  display:block; margin-top:8px;
  padding:12px; width:100%;
  box-sizing:border-box; border-radius:7px;
}
button {
  padding:12px 20px; margin-top:18px;
  background:#318ce7; color:white;
  border:0; border-radius:8px; cursor:pointer;
}
a {color:#77baff}
pre {white-space:pre-wrap;overflow-wrap:anywhere}
small {color:#b1c2d8}
</style>
</head>
<body>
<main>
<a href="/train">Back to Training</a>
<h1>Upload ML Model or Dataset</h1>

<div class="card">
  <label>File Type
    <select id="kind" onchange="changeType()">
      <option value="model">PyTorch Model (.py)</option>
      <option value="dataset">Dataset (.csv)</option>
    </select>
  </label>

  <label>Select File
    <input type="file" id="file" accept=".py">
  </label>

  <label>Cluster Authentication Token
    <input type="password" id="token" autocomplete="off">
  </label>

  <button onclick="uploadFile()">Upload and Analyze</button>

  <p><small>Only upload Python models you trust.
  Model code will execute when training starts.</small></p>
</div>

<div class="card">
  <h3>Analysis Result</h3>
  <pre id="result">No file uploaded.</pre>
</div>

<a href="/train">Go to Training Dashboard</a>
</main>

<script>
function changeType() {
  const kind = document.getElementById("kind").value;
  document.getElementById("file").accept =
    kind === "model" ? ".py" : ".csv";
  document.getElementById("file").value = "";
}

async function uploadFile() {
  const result = document.getElementById("result");
  const selected = document.getElementById("file").files[0];

  if (!selected) {
    result.textContent = "Select a file first.";
    return;
  }

  const body = new FormData();
  body.append("kind", document.getElementById("kind").value);
  body.append("file", selected);

  result.textContent = "Uploading...";

  try {
    const response = await fetch("/api/studio/upload", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " +
          document.getElementById("token").value
      },
      body
    });

    const data = await response.json();

    if (!response.ok) {
      result.textContent =
        "Upload failed: " + JSON.stringify(data);
      return;
    }

    result.textContent = JSON.stringify(data, null, 2)
      + "\\n\\nOpen Training Dashboard and refresh "
      + "to select the uploaded file.";

  } catch (error) {
    result.textContent = "Error: " + error;
  }
}
</script>
</body>
</html>
"""
