# Remediation Commands (2026-06-15)

## OSU reports

Exact local chunked gaps:

- `osu_scholars/osu_local_chunked_missing_pinecone.csv`
- `osu_scholars/osu_local_chunked_missing_mongo.csv`
- `osu_scholars/osu_local_chunked_missing_both.csv`
- `osu_scholars/osu_local_chunked_gap_summary.json`

Exact Excel-row chunking gaps:

- `osu_scholars/osu_missing_chunk_rows.csv`
- `osu_scholars/osu_missing_chunk_report_summary.json`

Pinecone diagnosis:

- `osu_scholars/osu_pinecone_diagnosis.json`

## OSU repair

Stage 1: scrape and chunk only the remaining OSU rows that do not already have local `chunks.json`.

```powershell
python fix_and_complete_osu.py --excel-path excel/OSU.xlsx --stages scrape --skip-pinecone-in-scrape --skip-mongodb-in-scrape
```

Stage 2: run the enrichment pass to improve profile quality and generate richer chunked output, Pinecone uploads, and Mongo syncs.

```powershell
python run_enrichment_pipeline.py --profiles-dir output/osu_faculty_run/profiles --output-dir output/osu_faculty_run --enable-cleaning
```

Stage 3: backfill any remaining local chunked OSU profiles into Pinecone and MongoDB.

```powershell
python fix_and_complete_osu.py --excel-path excel/OSU.xlsx --stages pinecone,mongo
```

Stage 4: verify final OSU coverage and quality.

```powershell
python pipeline_progress_report.py excel/OSU.xlsx --output-dir output/osu_faculty_run --chunking-output-dir output/osu_faculty_run/chunked_profiles
```

## Legends reports

- `legendary_scholars/legend_bad_profiles_rerun.csv`
- `legendary_scholars/legend_priority_rerun.csv`
- `legendary_scholars/legend_priority_rerun_slugs.txt`
- `legendary_scholars/legend_bad_profiles_summary.json`

## Legends repair

Re-run only the priority legend profiles.

```powershell
python run_legendary_enrichment.py --slugs-file legendary_scholars/legend_priority_rerun_slugs.txt --batch-size 20
```

Optional verification after the legend rerun:

```powershell
Get-Content legendary_scholars/legend_bad_profiles_summary.json
Get-Content legendary_scholars/legend_priority_rerun.csv | Select-Object -First 20
```
