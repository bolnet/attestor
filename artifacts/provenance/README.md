# Provenance artifacts

This directory holds public, verifiable proof of when each Attestor release
existed, independent of GitHub or PyPI. Combined with the SPDX headers
(`SPDX-FileCopyrightText` in every `attestor/**/*.py`) and the canary tokens
embedded in `attestor/__init__.py`, `attestor/_branding.py`,
`tests/conftest.py`, and `README.md`, these stamps form the original-author
record for the project.

## What's here

- `SHA256SUMS.txt` — content hashes of each PyPI release artifact.
- `*.ots` — OpenTimestamps proofs anchoring those hashes to the Bitcoin
  blockchain. Each `.ots` file is independently verifiable; nothing about
  this repo, GitHub, or PyPI is required.

## Verifying

```bash
pipx install opentimestamps-client      # one-time
cd artifacts/provenance

# Re-download the artifact from PyPI (or use a local copy)
pip download attestor==4.0.0 --no-deps --dest .

# Verify the SHA-256 matches what we stamped
shasum -a 256 -c SHA256SUMS.txt

# Upgrade the stamp (fills in the Bitcoin block proof once it has confirmed)
ots upgrade attestor-4.0.0-py3-none-any.whl.ots
ots upgrade attestor-4.0.0.tar.gz.ots

# Verify against the calendars
ots verify attestor-4.0.0-py3-none-any.whl.ots -f attestor-4.0.0-py3-none-any.whl
ots verify attestor-4.0.0.tar.gz.ots          -f attestor-4.0.0.tar.gz
```

A successful `ots verify` reports the earliest Bitcoin block-time at which
the file's hash was already known. That timestamp cannot be backdated.

## Adding a new release

```bash
# After publishing v4.x.y to PyPI:
cd artifacts/provenance
curl -sLO https://files.pythonhosted.org/.../attestor-4.x.y-py3-none-any.whl
curl -sLO https://files.pythonhosted.org/.../attestor-4.x.y.tar.gz
shasum -a 256 attestor-4.x.y* >> SHA256SUMS.txt
ots stamp attestor-4.x.y-py3-none-any.whl attestor-4.x.y.tar.gz
git add SHA256SUMS.txt *.ots
git commit -S -m "chore: stamp v4.x.y release artifacts"
```

The wheels and sdists themselves are not committed — PyPI is canonical.
Only the hashes and OTS proofs live here.
