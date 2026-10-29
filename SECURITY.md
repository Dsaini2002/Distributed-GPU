# Security Policy

**Distributed-GPU Studio is an experimental LAN-only beta.** It should not be exposed to the public internet or used to process sensitive data on an untrusted network.

## Supported environment

The current code is being validated on personally controlled Windows machines with NVIDIA CUDA GPUs. It is not hardened for multi-tenant or public-cloud deployment.

## Known limitations

- Host/worker service traffic currently uses HTTP over a trusted LAN, not end-to-end TLS.
- Some existing installations use short legacy cluster tokens; coordinated rotation is not yet shipped.
- Host-side approved-device state is currently not reliably persistent across service restarts.
- Unverified Windows installer outputs should not be described as production-safe.

## Reporting issues

Report non-sensitive bugs through GitHub Issues in this repository. For a security vulnerability, use a **private report** to the repository maintainer; do not publish working exploitation details or credentials in an Issue. Configure GitHub's *Private vulnerability reporting* in repository Settings → Security if you want to receive reports privately.

## Credential and data handling

- Never commit `.dgpu/`, `.env*`, `.venv/`, local training datasets, `*.pt`/`*.pth` model weights, bearer tokens or encrypted DPAPI credential files.
- Do not copy Windows DPAPI-protected blobs between accounts as a substitute for pairing.
- Do not port-forward ports `45400`, `45420` or `45430` to the internet.
- Check files with `git diff --cached --name-only` and `git diff --cached` before every push.
- If secrets were committed in the past, removing the file from the latest commit is insufficient: revoke/rotate the secret and remove it from Git history using appropriate tooling after backing up the repo.
