# External data and model artifacts

The Git repository contains only the current bakery web application, production
n8n outbound workflow, documentation, tests, template, and production model.

Large or generated assets are intentionally excluded from Git:

- raw and processed datasets under `data/`;
- training utilities, images, generated annotations, and review queues;
- experiment checkpoints, legacy ONNX/video models, and non-production weights;
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
