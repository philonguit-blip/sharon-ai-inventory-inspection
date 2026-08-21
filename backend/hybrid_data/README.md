# Hybrid Foundation data

This directory stores private/generated Foundation artifacts and is not a
source-code directory.

- `references/<PRODUCT_CODE>/`: approved top-down reference crops/images.
- `reference_embeddings.npz`: generated DINOv2 vectors.
- Product metadata is tracked in `backend/config/hybrid_reference_registry.json`.

Create reviewable original + white-background candidates without overwriting
the source folder:

```powershell
python scripts/prepare_foundation_references.py --source D:\raw\SKU --output D:\review\SKU --background both
```

Add `--input-mode tight` when `--source` already contains approved reference
crops and their current frame must remain unchanged.

Only copy visually approved outputs into `references/<PRODUCT_CODE>/`. Keep a
mix of original and white-background references unless runtime preprocessing is
also changed to white; a white-only bank can introduce a background-domain
shift.

Run `python scripts/hybrid_reference_manager.py --help` from the repository
root. Generated images and NPZ files are ignored by Git; back them up to the
controlled R2 artifact bucket before moving to another workstation.
