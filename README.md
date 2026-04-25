# open-plates

Structured terminal chart generator for flight-sim airfields. Takes a
declarative procedure definition (YAML) and renders a publication-quality
instrument approach plate as SVG, PDF, or PNG.

Built for DCS World pilots (fictional airbases with no published charts)
and MSFS bush-strip flyers (non-public fields with no IAP data).

**Not for real-world flight operations.**

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and
[Task](https://taskfile.dev/).

```bash
task setup          # install dependencies
task validate:all   # validate all example procedures against the schema
task render:example # render the UGSB ILS RWY 13 example to PDF
```

## Usage

### Validate a procedure

```bash
task validate -- examples/ugsb-ils-rwy-13.yaml
```

### Render a chart

```bash
# SVG
task render -- examples/etad-vor-dme-rwy-05.yaml out/etad-vor-dme-rwy-05.svg

# PDF
task render -- examples/ugsb-ils-rwy-13.yaml out/ugsb-ils-rwy-13.pdf

# PNG
task render -- examples/lszh-ils-z-rwy-14.yaml out/lszh-ils-z-rwy-14.png

# Render everything
task render:all
```

Output format is inferred from the file extension.

### Build basemap layers

Basemap scripts fetch geographic data (SRTM elevation, OpenStreetMap
features) and write assets that the renderer composites under the chart:

```bash
task basemap -- caucasus   # all layers for a region
task basemap:hillshade -- caucasus
task basemap:roads -- caucasus
task basemap:waterways -- caucasus
task basemap:obstacles -- caucasus
task basemap:populated -- caucasus
```

Downloaded source data caches under `data/cache/` (gitignored).

## Example procedures

| File | Description |
|------|-------------|
| `ugsb-ils-rwy-13.yaml` | ILS RWY 13 into Batumi (UGSB), DCS Caucasus |
| `etad-vor-dme-rwy-05.yaml` | VOR/DME arc approach to RWY 05, Spangdahlem AB (ETAD) |
| `etad-ils-rwy-23-straight.yaml` | ILS RWY 23 straight-in, Spangdahlem AB (ETAD) |
| `lszh-ils-z-rwy-14.yaml` | ILS Z RWY 14, Zurich (LSZH) |

All examples validate against `schema/procedure.schema.json`.

## Authoring procedures

Procedures are authored in YAML and validated against a JSON Schema
(draft 2020-12) that uses ARINC 424 path terminators as the leg
vocabulary (`IF`, `TF`, `CF`, `DF`, `CA`, `VA`, `AF`, `RF`, `HA`, `HM`,
etc.). See `schema/procedure.schema.json` for the full specification and
the example files for working templates.

## Testing

```bash
task test           # run the suite
task test:verbose   # verbose output
task test:watch     # re-run on file changes
task ci             # validate all examples + run tests
```

## License

MIT
