# Admin Portal

Next.js admin module for launching repo scripts through the FastAPI backend.

## Required backend

Start the FastAPI app from the repo root:

```bash
python start_server.py
```

The UI expects the backend at `http://localhost:8003` by default.

Override with:

```bash
set NEXT_PUBLIC_API_BASE_URL=http://localhost:8003
```

## Enterprise platform features

The admin script platform persists jobs, audit events, and schedules in MongoDB (`MONGODB_URI`).

- **Persistent jobs** — run history and logs survive API restarts
- **Worker pool** — queued execution with configurable concurrency (`ADMIN_MAX_CONCURRENT_JOBS`, default `2`)
- **Approval gates** — high-impact (`danger`) scripts require explicit approval (`ADMIN_APPROVAL_REQUIRED_RISKS`)
- **Audit trail** — immutable event log at `/api/admin/scripts/audit`
- **Schedules** — UTC cron schedules at `/api/admin/scripts/schedules`
- **Platform status** — queue depth and worker metrics at `/api/admin/scripts/platform/status`

## Start the UI

```bash
cd admin-portal
npm install
npm run dev
```

Open `http://localhost:3000/automation`.
