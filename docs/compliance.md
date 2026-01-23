# Licensing + Compliance Record (Task 1)

## Copernicus (Sentinel) usage conditions
- Copernicus products/services are provided under an open access policy; the operational policy is documented by Copernicus. :contentReference[oaicite:13]{index=13}
- The Copernicus programme’s legal basis includes Regulation (EU) No 377/2014 (historical; later replaced by newer regulation). :contentReference[oaicite:14]{index=14}
- This repository must retain a record of data sources and the applicable policy references for any Copernicus-derived products used.

## USGS (Landsat) usage conditions
- USGS Landsat data are generally public domain with no restriction on use/redistribution; USGS requests proper credit/acknowledgement as the data source. :contentReference[oaicite:15]{index=15}
- Any derived maps/products/publications must include the requested USGS credit line format (see USGS crediting guidance). :contentReference[oaicite:16]{index=16}

## Privacy constraint (aggregation level)
- Analyses and outputs are aggregated at village level to prevent inference about individuals. :contentReference[oaicite:17]{index=17}
- Administrative boundaries follow Statistics Bureau of Japan guidance (documented in the thesis plan); no person-level inference is produced. :contentReference[oaicite:18]{index=18}

## Operational safeguards
- `outputs/` contains only generated artifacts and is excluded from Git history (`.gitignore`).
- No credentials, tokens, or keys are committed to the repository.
