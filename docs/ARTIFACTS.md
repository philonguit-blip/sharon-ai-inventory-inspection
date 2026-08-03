# External data and model artifacts

The Git repository contains application source, n8n workflow definitions,
documentation, tests, templates, and the production backend models required by
the current bakery-counting service.

Large or generated assets are intentionally excluded from Git:

- raw and processed datasets under `data/`;
- training images, generated annotations, and review queues;
- experiment checkpoints and non-production model weights;
- demo videos, runtime jobs, logs, generated reports, and R2 artifacts;
- Python virtual environments and local Cloudflare state.

Keep these assets in the approved R2 bucket or another versioned internal
artifact store. Record the object path, checksum, model version, dataset version,
and training configuration whenever promoting a new production model.

The production model expected by the current backend is:

```text
backend/models/best_YOLO26s_PROD_1_SKU_v4.pt
```

Before replacing it, run the backend unit tests, validate the confidence
thresholds, and test a representative bakery image set.
