# Unified Pipeline Documentation

## Overview

The `unified_pipeline.py` script provides a complete end-to-end pipeline that processes Excel files containing profile URLs and creates chunked JSON files ready for vector database storage.

## Pipeline Flow

```
Excel File (with profile URLs)
    ↓
[Step 1] Read Excel & Extract URLs
    ↓
[Step 2] Scrape Profile URLs (using Playwright)
    - Extract profile page content
    - Download and process documents (PDFs, Word docs)
    - Extract linked webpage content
    ↓
[Step 3] Clean Extracted Text
    - Remove noise and formatting issues
    - Normalize text
    - Basic text cleaning
    ↓
[Step 4] Create Section-Aware Chunks (Optional)
    - Use LLM to identify sections (Biography, Education, etc.)
    - Create semantic chunks with proper overlap
    - Save to chunks.json format
    ↓
[Step 5] Save Profile JSON
    - Save cleaned text and metadata
    - Link to chunks.json if available
    - Store in output/profiles/{profile_id}/
```

## Usage

### Basic Usage

```bash
python unified_pipeline.py path/to/profiles.xlsx
```

### With Options

```bash
# Disable LLM chunking (faster, less accurate)
python unified_pipeline.py profiles.xlsx --no-llm-chunking

# Use OpenAI for chunking instead of Ollama
python unified_pipeline.py profiles.xlsx --llm-provider openai --llm-model gpt-4o-mini

# Process only first 10 profiles
python unified_pipeline.py profiles.xlsx --limit 10

# Resume from profile 50
python unified_pipeline.py profiles.xlsx --start-from 50

# Custom output directories
python unified_pipeline.py profiles.xlsx --output-dir custom_output --chunking-output-dir custom_output/chunks
```

### URL List (Single Profile)

If you have multiple URLs that belong to one profile, put them in a `.txt` file
with one URL per line and run:

```bash
python unified_pipeline.py --urls-file path/to/urls.txt --profile-name "Jane Doe"
```

Optional:

```bash
# Explicitly set the primary profile_url stored in output JSON
python unified_pipeline.py --urls-file path/to/urls.txt --profile-url https://example.edu/profile
```

By default, URLs-file mode writes output under:

```
output/url_list_runs/<timestamp>/
```

You can override the output folder:

```bash
python unified_pipeline.py --urls-file path/to/urls.txt --output-dir output/custom_run
```

## Excel File Format

Your Excel file must contain:
- **Required column**: `source` or `profile_url` (contains profile URLs)
- **Optional columns**: `name`, `email`, `university`, `department`, etc. (preserved in output)

Example:
| source | name | university |
|--------|------|------------|
| https://example.edu/profile1 | John Doe | Example University |
| https://example.edu/profile2 | Jane Smith | Example University |

## Output Structure

```
output/
├── profiles/
│   ├── {profile_id_1}/
│   │   ├── {profile_id_1}.json
│   │   ├── source_chunks.json
│   │   └── claims.json
│   ├── {profile_id_2}/
│   │   ├── {profile_id_2}.json
│   │   ├── source_chunks.json
│   │   └── claims.json
│   └── ...
└── chunked_profiles/
    ├── {profile_id_1}/
    │   └── chunks.json
    ├── {profile_id_2}/
    │   └── chunks.json
    └── ...
└── source_registry.jsonl
```

### Profile JSON Structure

```json
{
  "profile_id": "uuid",
  "name": "Professor Name",
  "profile_url": "https://...",
  "all_urls": ["https://...", "https://..."],
  "raw_text": "Combined raw text from all sources",
  "clean_text": "Cleaned and normalized text",
  "has_cv": true,
  "chunks_available": true,
  "chunks_file": "chunked_profiles/{profile_id}/chunks.json",
  "source_chunks_file": "profiles/{profile_id}/source_chunks.json",
  "claims_file": "profiles/{profile_id}/claims.json",
  "source_registry": "output/source_registry.jsonl",
  "created_at": "2025-12-03T...",
  "updated_at": "2025-12-03T..."
}
```

### Source Registry (JSONL)

Each line is a `SourceRecord` with license/policy metadata, fetch metadata, and quality/PII flags.

### Source Chunks (per profile)

`source_chunks.json` contains per-source chunk records with `source_id`, `allowed_use`, and offsets for defensible provenance.

### Chunks JSON Structure

```json
{
  "profile_id": "uuid",
  "sections": {
    "Biography": [
      {
        "profile_id": "uuid",
        "section": "Biography",
        "chunk_id": "uuid",
        "order": 0,
        "text": "Chunk text content..."
      }
    ],
    "Education": [...],
    "Research Interests": [...]
  }
}
```

## Configuration

### LLM Chunking

- **Enabled by default**: Uses LLM to identify sections and create semantic chunks
- **Provider options**: `ollama` (default) or `openai`
- **Ollama models**: `mistral:7b`, `llama3:8b`, etc.
- **OpenAI models**: `gpt-4o-mini`, `gpt-4`, etc.

### Environment Variables

Required in `.env` file:

```env
# For OpenAI (if using OpenAI chunking)
OPENAI_API_KEY=your-key-here

# For Ollama (if using Ollama chunking)
# Ollama should be running at http://localhost:11434
```

## Performance

- **Scraping**: ~5-15 seconds per profile (depends on page complexity)
- **Cleaning**: ~1-2 seconds per profile
- **Chunking**: ~10-30 seconds per profile (if LLM chunking enabled)
- **Total**: ~15-50 seconds per profile

## Error Handling

- Failed scrapes are logged but don't stop the pipeline
- Profiles with no content are marked as failed
- Chunking failures fall back to saving profile without chunks
- All errors are logged with details

## Next Steps

After running the pipeline:

1. **Upload to Pinecone**: Use `upload_chunks_to_pinecone.py` to upload chunks to vector database
2. **Sync to MongoDB**: Use `sync_profiles_to_mongodb.py` to create MongoDB documents
   - Optional raw backup: `sync_vectordb_to_mongodb.py` mirrors Pinecone chunk records into MongoDB (`vector_chunks` collection)
3. **Review Output**: Check `output/profiles/` and `output/chunked_profiles/` for results

## Troubleshooting

### Common Issues

1. **"No valid URLs found"**
   - Check Excel file has `source` or `profile_url` column
   - Ensure URLs are not empty

2. **"Scraping failed"**
   - Check internet connection
   - Verify URLs are accessible
   - Check Playwright is installed: `playwright install`

3. **"LLM chunking failed"**
   - If using Ollama: Ensure Ollama is running
   - If using OpenAI: Check API key is set
   - Use `--no-llm-chunking` to skip chunking

4. **"Import errors"**
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure all services are in correct directories

## Related Scripts

- `upload_chunks_to_pinecone.py`: Upload chunks to Pinecone vector database
- `sync_profiles_to_mongodb.py`: Sync profiles to MongoDB with LLM summaries

## Optional Semantic Splitter

If you want to re-enable semantic splitting later, see:

- `semantic_splitter.py`: standalone semantic chunking helper (not wired by default)

## Cleaning Existing Chunks

To post-process existing `chunks.json` files and remove boilerplate/short chunks:

```bash
python clean_existing_chunks.py --chunks-root output/url_list_runs/<timestamp>/chunked_profiles
```

For a single file:

```bash
python clean_existing_chunks.py --chunks-file path/to/chunks.json --profile-name "Person Name"
```
- `delete_pinecone_data.py`: Delete all data from Pinecone
- `start_server.py`: Start FastAPI server for web interface
