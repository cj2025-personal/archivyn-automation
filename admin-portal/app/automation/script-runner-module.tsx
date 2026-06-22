"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState, useTransition } from "react";

type ModuleField = {
  id: string;
  label: string;
  type: "text" | "number" | "textarea" | "checkbox" | "select";
  help_text: string;
  default: unknown;
  placeholder: string;
  required: boolean;
  options: Array<{ label: string; value: string }>;
};

type ScriptModule = {
  id: string;
  script_id: string;
  title: string;
  subtitle: string;
  summary: string;
  category: string;
  scope: string;
  risk: "safe" | "caution" | "danger";
  mode: "structured" | "raw";
  featured: boolean;
  run_label: string;
  success_message: string;
  caution_message: string;
  indications: string[];
  fields: ModuleField[];
  usage_examples: string[];
  script_filename: string;
  script_path: string;
};

type ScriptJob = {
  id: string;
  script_id: string;
  script_filename: string;
  command: string[];
  raw_args: string;
  module_id?: string | null;
  risk?: string;
  approval_required?: boolean;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  log_count: number;
  log_version: number;
  logs: string[];
  error?: string | null;
  schedule_id?: string | null;
};

type ScriptSchedule = {
  id: string;
  name: string;
  module_id: string;
  values: ModuleValues;
  cron_expression: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  last_run_at?: string | null;
  last_job_id?: string | null;
  last_status?: string | null;
};

type AuditEvent = {
  id: string;
  event_type: string;
  created_at: string;
  job_id?: string | null;
  module_id?: string | null;
  script_id?: string | null;
  risk?: string | null;
  status?: string | null;
  message?: string | null;
};

type PlatformStatus = {
  max_concurrent_jobs: number;
  active_workers: number;
  running_jobs: number;
  queued_jobs: number;
  pending_approval_jobs: number;
  queue_depth: number;
};

type WorkspaceView = "run" | "schedules" | "audit";

type BannerState = {
  tone: "success" | "warning" | "danger";
  message: string;
} | null;

type ModuleValues = Record<string, unknown>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8003";

const NAV_LINKS = [
  {
    href: "/automation",
    label: "Automation Modules",
    description: "Run scripts with structured admin controls.",
    internal: true,
    icon: "grid",
  },
];

function riskLabel(risk: ScriptModule["risk"]) {
  if (risk === "danger") {
    return "High Impact";
  }
  if (risk === "caution") {
    return "Needs Care";
  }
  return "Routine";
}

function statusTone(status: string) {
  if (status === "completed") {
    return "success";
  }
  if (status === "failed" || status === "rejected") {
    return "danger";
  }
  if (status === "running" || status === "terminating") {
    return "active";
  }
  if (status === "pending_approval") {
    return "active";
  }
  if (status === "queued") {
    return "muted";
  }
  return "muted";
}

function formatEventType(value: string) {
  return value.replaceAll("_", " ");
}

function moduleDefaults(module: ScriptModule): ModuleValues {
  const values: ModuleValues = {};
  for (const field of module.fields) {
    values[field.id] = field.default;
  }
  return values;
}

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "pending";
  }
  return value.replace("T", " ").slice(0, 19);
}

function valueAsString(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function isBlankValue(value: unknown) {
  if (typeof value === "boolean") {
    return false;
  }
  return valueAsString(value).trim() === "";
}

function NavIcon({ icon }: { icon: string }) {
  if (icon === "database") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <ellipse cx="12" cy="6" rx="7" ry="3.2" />
        <path d="M5 6v6c0 1.8 3.1 3.2 7 3.2s7-1.4 7-3.2V6" />
        <path d="M5 12v6c0 1.8 3.1 3.2 7 3.2s7-1.4 7-3.2v-6" />
      </svg>
    );
  }
  if (icon === "folder") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 7.5h6l1.8 2H21v7.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        <path d="M3 7.5V6a2 2 0 0 1 2-2h4l1.7 2H19a2 2 0 0 1 2 2v1.5" />
      </svg>
    );
  }
  if (icon === "spark") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3l1.7 5.1L19 10l-5.3 1.9L12 17l-1.7-5.1L5 10l5.3-1.9z" />
        <path d="M18.5 3.5l.7 2 .8.3-.8.3-.7 2-.7-2-.8-.3.8-.3z" />
        <path d="M5.5 14.5l.9 2.4 2.4.9-2.4.9-.9 2.4-.9-2.4-2.4-.9 2.4-.9z" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />
    </svg>
  );
}

function ModuleIcon({ category, mode }: { category: string; mode: ScriptModule["mode"] }) {
  const normalized = category.toLowerCase();
  if (normalized.includes("editorial")) {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 5.5h9a2 2 0 0 1 2 2v11H7a2 2 0 0 0-2 2z" />
        <path d="M16 7.5h3a1 1 0 0 1 1 1V18a2 2 0 0 1-2 2h-1z" />
        <path d="M8.5 10h5M8.5 13h5M8.5 16h3.5" />
      </svg>
    );
  }
  if (normalized.includes("universit")) {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 9.5L12 4l9 5.5-9 5.5z" />
        <path d="M6.5 11.5V16c0 1.8 2.5 3 5.5 3s5.5-1.2 5.5-3v-4.5" />
      </svg>
    );
  }
  if (normalized.includes("legend")) {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 4l1.8 3.9 4.2.6-3 3 0.7 4.1-3.7-2.1-3.7 2.1.7-4.1-3-3 4.2-.6z" />
        <path d="M8 18.5h8" />
      </svg>
    );
  }
  if (mode === "raw") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4.5 6.5h15v11h-15z" />
        <path d="M7.5 10l2 2-2 2M11.5 14h4" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6h16v12H4z" />
      <path d="M8 10h8M8 14h5" />
    </svg>
  );
}

export function AutomationModule() {
  const [modules, setModules] = useState<ScriptModule[]>([]);
  const [jobs, setJobs] = useState<ScriptJob[]>([]);
  const [activeModuleId, setActiveModuleId] = useState("");
  const [activeJob, setActiveJob] = useState<ScriptJob | null>(null);
  const [moduleValues, setModuleValues] = useState<Record<string, ModuleValues>>({});
  const [moduleFieldErrors, setModuleFieldErrors] = useState<Record<string, string>>({});
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [banner, setBanner] = useState<BannerState>(null);
  const [search, setSearch] = useState("");
  const [scopeFilter, setScopeFilter] = useState("All");
  const [categoryFilter, setCategoryFilter] = useState("All");
  const [riskFilter, setRiskFilter] = useState("All");
  const [catalogTab, setCatalogTab] = useState<"featured" | "advanced">("featured");
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("run");
  const [schedules, setSchedules] = useState<ScriptSchedule[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [platformStatus, setPlatformStatus] = useState<PlatformStatus | null>(null);
  const [scheduleName, setScheduleName] = useState("");
  const [scheduleCron, setScheduleCron] = useState("30 1 * * *");
  const [isPending, startTransition] = useTransition();
  const deferredSearch = useDeferredValue(search);
  const completionMarker = useRef("");
  const launchedByJobId = useRef<Record<string, string>>({});

  useEffect(() => {
    startTransition(() => {
      void loadModules();
      void loadJobs();
      void loadSchedules();
      void loadAudit();
      void loadPlatformStatus();
    });
  }, []);

  useEffect(() => {
    if (!activeJob?.id || activeJob.status === "pending_approval" || activeJob.status === "rejected") {
      return;
    }
    const streamUrl = `${API_BASE}/api/admin/scripts/jobs/${activeJob.id}/stream`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as { job: Omit<ScriptJob, "logs">; logs: string[] };
        const nextJob = { ...payload.job, logs: payload.logs } as ScriptJob;
        setActiveJob(nextJob);
        setJobs((previous) => {
          const withoutCurrent = previous.filter((job) => job.id !== nextJob.id);
          return [nextJob, ...withoutCurrent].slice(0, 16);
        });
        setStreamError(null);
        if (nextJob.status === "completed" || nextJob.status === "failed" || nextJob.status === "rejected") {
          eventSource.close();
        }
      } catch {
        setStreamError("Received malformed stream payload.");
      }
    };

    eventSource.onerror = () => {
      setStreamError("Live stream disconnected. Refresh jobs if the process is still running.");
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [activeJob?.id]);

  const scopes = ["All", ...new Set(modules.map((module) => module.scope))];
  const categories = ["All", ...new Set(modules.map((module) => module.category))];

  const filteredModules = modules.filter((module) => {
    const haystack = `${module.title} ${module.subtitle} ${module.summary} ${module.category} ${module.scope}`.toLowerCase();
    if (deferredSearch && !haystack.includes(deferredSearch.toLowerCase())) {
      return false;
    }
    if (scopeFilter !== "All" && module.scope !== scopeFilter) {
      return false;
    }
    if (categoryFilter !== "All" && module.category !== categoryFilter) {
      return false;
    }
    if (riskFilter !== "All" && module.risk !== riskFilter.toLowerCase()) {
      return false;
    }
    return true;
  });

  const featuredModules = filteredModules.filter((module) => module.featured);
  const advancedModules = filteredModules.filter((module) => !module.featured);

  const activeModule =
    modules.find((module) => module.id === activeModuleId) ??
    (filteredModules.length > 0 ? filteredModules[0] : modules[0] ?? null);

  const activeModuleInputFields = useMemo(
    () => activeModule?.fields.filter((field) => field.type !== "checkbox") ?? [],
    [activeModule],
  );
  const activeModuleToggleFields = useMemo(
    () => activeModule?.fields.filter((field) => field.type === "checkbox") ?? [],
    [activeModule],
  );

  useEffect(() => {
    if (!activeModuleId && modules.length > 0) {
      setActiveModuleId(modules[0].id);
      return;
    }
    if (activeModuleId && !modules.some((module) => module.id === activeModuleId) && modules.length > 0) {
      setActiveModuleId(modules[0].id);
    }
  }, [activeModuleId, modules]);

  useEffect(() => {
    if (!activeModule) {
      return;
    }
    setModuleValues((previous) => {
      if (previous[activeModule.id]) {
        return previous;
      }
      return {
        ...previous,
        [activeModule.id]: moduleDefaults(activeModule),
      };
    });
  }, [activeModule]);

  useEffect(() => {
    if (!activeJob) {
      return;
    }
    const module = moduleForJob(activeJob);
    const marker = `${activeJob.id}:${activeJob.status}:${activeJob.exit_code ?? "pending"}`;
    if (completionMarker.current === marker) {
      return;
    }
    if (activeJob.status === "completed") {
      setBanner({
        tone: "success",
        message: module?.success_message ?? "Script completed successfully.",
      });
      completionMarker.current = marker;
    } else if (activeJob.status === "failed" || activeJob.status === "rejected") {
      setBanner({
        tone: "danger",
        message: activeJob.error || `The script exited with code ${activeJob.exit_code ?? "unknown"}. Review the console output.`,
      });
      completionMarker.current = marker;
    }
  }, [activeJob, modules]);

  function moduleForJob(job: ScriptJob | null) {
    if (!job) {
      return null;
    }
    const mappedModuleId = launchedByJobId.current[job.id];
    if (mappedModuleId) {
      return modules.find((module) => module.id === mappedModuleId) ?? null;
    }
    return modules.find((module) => module.script_id === job.script_id) ?? null;
  }

  async function loadModules() {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/modules`);
      if (!response.ok) {
        throw new Error(`Modules request failed with ${response.status}`);
      }
      const payload = (await response.json()) as { modules: ScriptModule[] };
      setModules(payload.modules);
      setModuleValues((previous) => {
        const next = { ...previous };
        for (const module of payload.modules) {
          if (!next[module.id]) {
            next[module.id] = moduleDefaults(module);
          }
        }
        return next;
      });
      setCatalogError(null);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : "Unable to load admin modules.");
    }
  }

  async function refreshModules() {
    startTransition(() => {
      void (async () => {
        try {
          const response = await fetch(`${API_BASE}/api/admin/scripts/modules/refresh`, {
            method: "POST",
          });
          if (!response.ok) {
            throw new Error(`Refresh failed with ${response.status}`);
          }
          const payload = (await response.json()) as { modules: ScriptModule[] };
          setModules(payload.modules);
          setModuleValues((previous) => {
            const next = { ...previous };
            for (const module of payload.modules) {
              if (!next[module.id]) {
                next[module.id] = moduleDefaults(module);
              }
            }
            return next;
          });
          setCatalogError(null);
        } catch (error) {
          setCatalogError(error instanceof Error ? error.message : "Unable to refresh admin modules.");
        }
      })();
    });
  }

  async function loadPlatformStatus() {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/platform/status`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as { platform: PlatformStatus };
      setPlatformStatus(payload.platform);
    } catch {
      // Optional telemetry.
    }
  }

  async function loadSchedules() {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/schedules`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as { schedules: ScriptSchedule[] };
      setSchedules(payload.schedules);
    } catch {
      // Schedules are optional when persistence is unavailable.
    }
  }

  async function loadAudit() {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/audit`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as { events: AuditEvent[] };
      setAuditEvents(payload.events);
    } catch {
      // Audit is optional when persistence is unavailable.
    }
  }

  async function loadJobs() {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/jobs`);
      if (!response.ok) {
        throw new Error(`Jobs request failed with ${response.status}`);
      }
      const payload = (await response.json()) as { jobs: ScriptJob[] };
      setJobs(payload.jobs);
      void loadPlatformStatus();
    } catch {
      // Keep the module usable even if history fails.
    }
  }

  function setModuleField(moduleId: string, fieldId: string, value: unknown) {
    setModuleValues((previous) => ({
      ...previous,
      [moduleId]: {
        ...(previous[moduleId] ?? {}),
        [fieldId]: value,
      },
    }));
    setModuleFieldErrors((previous) => {
      if (!previous[fieldId]) {
        return previous;
      }
      const next = { ...previous };
      delete next[fieldId];
      return next;
    });
  }

  function validateModule(module: ScriptModule, values: ModuleValues) {
    const errors: Record<string, string> = {};
    for (const field of module.fields) {
      if (field.required && isBlankValue(values[field.id])) {
        errors[field.id] = `${field.label} is required.`;
      }
    }

    if (module.id === "legend_readiness_audit") {
      const hasNames = valueAsString(values.names).trim() !== "";
      const hasProfileIds = valueAsString(values.profile_ids).trim() !== "";
      const auditAll = Boolean(values.audit_all);
      if (!hasNames && !hasProfileIds && !auditAll) {
        errors.names = "Provide legend names, profile ids, or enable full collection audit.";
      }
    }

    if (module.id === "daily_story_worker" && Boolean(values.enforce_profile_quality) && Boolean(values.ignore_profile_quality)) {
      errors.ignore_profile_quality = "Choose either enforce or ignore profile quality, not both.";
    }

    return errors;
  }

  async function launchModule() {
    if (!activeModule) {
      return;
    }
    const values = moduleValues[activeModule.id] ?? moduleDefaults(activeModule);
    const errors = validateModule(activeModule, values);
    setModuleFieldErrors(errors);
    setBanner(null);
    setStreamError(null);

    if (Object.keys(errors).length > 0) {
      setBanner({
        tone: "warning",
        message: "Fix the highlighted fields before launching the module.",
      });
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/jobs/from-module`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          module_id: activeModule.id,
          values,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: "Launch failed." }));
        throw new Error(detail.detail || `Launch failed with ${response.status}`);
      }
      const payload = (await response.json()) as { job: ScriptJob };
      launchedByJobId.current[payload.job.id] = activeModule.id;
      setActiveJob(payload.job);
      setJobs((previous) => [payload.job, ...previous.filter((job) => job.id !== payload.job.id)].slice(0, 50));
      if (payload.job.status === "pending_approval") {
        setBanner({
          tone: "warning",
          message: "High-impact script queued for approval. Approve it below to start execution.",
        });
      } else {
        setBanner({
          tone: "success",
          message: `${activeModule.title} started successfully. Live logs are streaming below.`,
        });
      }
      void loadAudit();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to start module.",
      });
    }
  }

  async function approveJob(jobId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/jobs/${jobId}/approve`, { method: "POST" });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: "Approval failed." }));
        throw new Error(detail.detail || `Approval failed with ${response.status}`);
      }
      const payload = (await response.json()) as { job: ScriptJob };
      setActiveJob(payload.job);
      setJobs((previous) => [payload.job, ...previous.filter((job) => job.id !== payload.job.id)].slice(0, 50));
      setBanner({ tone: "success", message: "Job approved and queued for execution." });
      void loadAudit();
      void loadPlatformStatus();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to approve job.",
      });
    }
  }

  async function rejectJob(jobId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/jobs/${jobId}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "Rejected from automation console." }),
      });
      if (!response.ok) {
        throw new Error(`Reject failed with ${response.status}`);
      }
      const payload = (await response.json()) as { job: ScriptJob };
      setActiveJob(payload.job);
      setJobs((previous) => [payload.job, ...previous.filter((job) => job.id !== payload.job.id)].slice(0, 50));
      setBanner({ tone: "warning", message: "Job rejected before execution." });
      void loadAudit();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to reject job.",
      });
    }
  }

  async function createSchedule() {
    if (!activeModule) {
      return;
    }
    const values = moduleValues[activeModule.id] ?? moduleDefaults(activeModule);
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/schedules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: scheduleName.trim() || activeModule.title,
          module_id: activeModule.id,
          values,
          cron_expression: scheduleCron.trim(),
          enabled: true,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: "Schedule creation failed." }));
        throw new Error(detail.detail || `Schedule creation failed with ${response.status}`);
      }
      setBanner({ tone: "success", message: "Schedule created. Cron runs use UTC." });
      setScheduleName("");
      void loadSchedules();
      void loadAudit();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to create schedule.",
      });
    }
  }

  async function toggleSchedule(schedule: ScriptSchedule) {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/schedules/${schedule.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !schedule.enabled }),
      });
      if (!response.ok) {
        throw new Error(`Schedule update failed with ${response.status}`);
      }
      void loadSchedules();
      void loadAudit();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to update schedule.",
      });
    }
  }

  async function deleteSchedule(scheduleId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/schedules/${scheduleId}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Schedule delete failed with ${response.status}`);
      }
      void loadSchedules();
      void loadAudit();
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to delete schedule.",
      });
    }
  }

  async function terminateJob(jobId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/admin/scripts/jobs/${jobId}/terminate`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`Terminate failed with ${response.status}`);
      }
      const payload = (await response.json()) as { job: ScriptJob };
      setActiveJob(payload.job);
      setJobs((previous) => [payload.job, ...previous.filter((job) => job.id !== payload.job.id)].slice(0, 16));
      setBanner({
        tone: "warning",
        message: "Termination signal sent to the active process.",
      });
    } catch (error) {
      setBanner({
        tone: "danger",
        message: error instanceof Error ? error.message : "Unable to stop job.",
      });
    }
  }

  function renderField(module: ScriptModule, field: ModuleField) {
    const values = moduleValues[module.id] ?? {};
    const currentValue = values[field.id] ?? field.default;
    const error = moduleFieldErrors[field.id];

    if (field.type === "checkbox") {
      return (
        <label key={field.id} className={`toggle-field ${error ? "toggle-field--error" : ""}`}>
          <input
            type="checkbox"
            checked={Boolean(currentValue)}
            onChange={(event) => setModuleField(module.id, field.id, event.target.checked)}
          />
          <span>
            <strong>{field.label}</strong>
            <small>{field.help_text}</small>
            {error ? <em>{error}</em> : null}
          </span>
        </label>
      );
    }

    if (field.type === "textarea") {
      return (
        <label key={field.id} className="form-field">
          <span className="field-title">{field.label}</span>
          <textarea
            className={`field-input field-input--textarea ${error ? "field-input--error" : ""}`}
            value={valueAsString(currentValue)}
            placeholder={field.placeholder}
            onChange={(event) => setModuleField(module.id, field.id, event.target.value)}
          />
          <small>{error ?? field.help_text}</small>
        </label>
      );
    }

    if (field.type === "select") {
      return (
        <label key={field.id} className="form-field">
          <span className="field-title">{field.label}</span>
          <select
            className={`field-input ${error ? "field-input--error" : ""}`}
            value={valueAsString(currentValue)}
            onChange={(event) => setModuleField(module.id, field.id, event.target.value)}
          >
            {field.options.map((option) => (
              <option key={`${field.id}-${option.value}`} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <small>{error ?? field.help_text}</small>
        </label>
      );
    }

    return (
      <label key={field.id} className="form-field">
        <span className="field-title">{field.label}</span>
        <input
          className={`field-input ${error ? "field-input--error" : ""}`}
          type={field.type === "number" ? "number" : "text"}
          value={valueAsString(currentValue)}
          placeholder={field.placeholder}
          onChange={(event) => setModuleField(module.id, field.id, event.target.value)}
        />
        <small>{error ?? field.help_text}</small>
      </label>
    );
  }

  function selectModule(moduleId: string) {
    setActiveModuleId(moduleId);
    setModuleFieldErrors({});
    setBanner(null);
  }

  return (
    <div className="portal-shell">
      <aside className="portal-sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <span className="brand-mark__icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />
              </svg>
            </span>
            <div className="brand-mark__text">
              <strong>Automation</strong>
              <span>Admin Portal</span>
            </div>
          </div>
        </div>

        <nav className="module-nav" aria-label="Portal navigation">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              className={`module-link ${link.internal ? "module-link--current" : ""}`}
              href={link.href}
              target={link.internal ? undefined : "_blank"}
              rel={link.internal ? undefined : "noreferrer"}
              title={link.description}
            >
              <span className="module-link__icon">
                <NavIcon icon={link.icon} />
              </span>
              <span className="module-link__body">
                <strong>{link.label}</strong>
                <small>{link.description}</small>
              </span>
            </a>
          ))}
        </nav>

        <section className="sidebar-section">
          <div className="sidebar-heading">
            <span>Recent Jobs</span>
            <button className="ghost-button" onClick={() => void loadJobs()} type="button">
              Refresh
            </button>
          </div>
          <div className="job-list">
            {jobs.length === 0 ? (
              <p className="empty-copy" style={{ padding: "12px 8px", textAlign: "left" }}>
                No jobs yet
              </p>
            ) : (
              jobs.map((job) => (
                <button
                  key={job.id}
                  className={`job-pill job-pill--${statusTone(job.status)} ${
                    activeJob?.id === job.id ? "job-pill--selected" : ""
                  }`}
                  onClick={() => setActiveJob(job)}
                  type="button"
                >
                  <strong>{job.script_filename}</strong>
                  <span>{job.status}</span>
                  <small>{formatTimestamp(job.created_at)}</small>
                </button>
              ))
            )}
          </div>
        </section>
      </aside>

      <main className="portal-main">
        <header className="page-header">
          <div className="page-header__copy">
            <p className="eyebrow">Script Operations</p>
            <h2>Automation Console</h2>
            <p>Launch curated workflows, monitor live output, and manage script jobs from one place.</p>
          </div>
          <div className="page-header__actions">
            <div className="stat-row">
              <div className="stat-chip">
                <strong>{featuredModules.length}</strong>
                <span>Featured</span>
              </div>
              <div className="stat-chip">
                <strong>{advancedModules.length}</strong>
                <span>Advanced</span>
              </div>
              <div className="stat-chip">
                <strong>{platformStatus?.running_jobs ?? 0}</strong>
                <span>Running</span>
              </div>
              <div className="stat-chip">
                <strong>{platformStatus?.pending_approval_jobs ?? 0}</strong>
                <span>Approval</span>
              </div>
              <div className="stat-chip">
                <strong>{jobs.length}</strong>
                <span>Jobs</span>
              </div>
            </div>
            <button className="secondary-button" onClick={refreshModules} type="button" disabled={isPending}>
              {isPending ? "Refreshing…" : "Refresh"}
            </button>
            <a className="secondary-button secondary-button--link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
              API Docs
            </a>
          </div>
        </header>

        <div className="workspace">
          <div className="workspace-tabs" role="tablist">
            <button
              className={`workspace-tab ${workspaceView === "run" ? "workspace-tab--active" : ""}`}
              onClick={() => setWorkspaceView("run")}
              type="button"
            >
              Run Modules
            </button>
            <button
              className={`workspace-tab ${workspaceView === "schedules" ? "workspace-tab--active" : ""}`}
              onClick={() => setWorkspaceView("schedules")}
              type="button"
            >
              Schedules ({schedules.length})
            </button>
            <button
              className={`workspace-tab ${workspaceView === "audit" ? "workspace-tab--active" : ""}`}
              onClick={() => {
                setWorkspaceView("audit");
                void loadAudit();
              }}
              type="button"
            >
              Audit Log
            </button>
          </div>

          {workspaceView === "schedules" ? (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3>Scheduled Runs</h3>
                  <span className="card-subtitle">Cron expressions run in UTC. Scheduled danger scripts are pre-approved.</span>
                </div>
                <button className="secondary-button" onClick={() => void loadSchedules()} type="button">
                  Refresh
                </button>
              </div>
              <div className="card-body">
                {activeModule ? (
                  <section className="detail-section schedule-form">
                    <div className="detail-section__header">
                      <h4>Schedule current module</h4>
                      <span>{activeModule.title}</span>
                    </div>
                    <div className="form-grid">
                      <label className="form-field">
                        <span className="field-title">Schedule name</span>
                        <input
                          className="field-input"
                          value={scheduleName}
                          placeholder={activeModule.title}
                          onChange={(event) => setScheduleName(event.target.value)}
                        />
                      </label>
                      <label className="form-field">
                        <span className="field-title">Cron (UTC)</span>
                        <input
                          className="field-input"
                          value={scheduleCron}
                          onChange={(event) => setScheduleCron(event.target.value)}
                        />
                        <small>Example: `30 1 * * *` runs daily at 01:30 UTC.</small>
                      </label>
                    </div>
                    <div className="action-row">
                      <button className="primary-button" onClick={() => void createSchedule()} type="button">
                        Create schedule
                      </button>
                    </div>
                  </section>
                ) : null}

                {schedules.length === 0 ? (
                  <div className="empty-state">
                    <p>No schedules yet. Select a module under Run Modules, then create a cron schedule here.</p>
                  </div>
                ) : (
                  <div className="schedule-list">
                    {schedules.map((schedule) => {
                      const module = modules.find((item) => item.id === schedule.module_id);
                      return (
                        <div key={schedule.id} className="schedule-row">
                          <div className="schedule-row__body">
                            <strong>{schedule.name}</strong>
                            <small>{module?.title ?? schedule.module_id}</small>
                            <code>{schedule.cron_expression}</code>
                            <span>
                              Last run: {formatTimestamp(schedule.last_run_at)} · Status: {schedule.last_status ?? "never"}
                            </span>
                          </div>
                          <div className="action-row">
                            <button className="secondary-button" onClick={() => void toggleSchedule(schedule)} type="button">
                              {schedule.enabled ? "Disable" : "Enable"}
                            </button>
                            <button className="danger-button" onClick={() => void deleteSchedule(schedule.id)} type="button">
                              Delete
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : null}

          {workspaceView === "audit" ? (
            <div className="card">
              <div className="card-header">
                <div>
                  <h3>Audit Trail</h3>
                  <span className="card-subtitle">Immutable record of launches, approvals, schedules, and job outcomes</span>
                </div>
                <button className="secondary-button" onClick={() => void loadAudit()} type="button">
                  Refresh
                </button>
              </div>
              <div className="card-body">
                {auditEvents.length === 0 ? (
                  <div className="empty-state">
                    <p>No audit events recorded yet.</p>
                  </div>
                ) : (
                  <div className="audit-list">
                    {auditEvents.map((event) => (
                      <div key={event.id} className="audit-row">
                        <div className="audit-row__meta">
                          <strong>{formatEventType(event.event_type)}</strong>
                          <span>{formatTimestamp(event.created_at)}</span>
                        </div>
                        <p>
                          {event.message ||
                            [event.module_id, event.script_id, event.status].filter(Boolean).join(" · ") ||
                            "—"}
                        </p>
                        {event.job_id ? <code>{event.job_id.slice(0, 12)}…</code> : null}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : null}

          {workspaceView === "run" ? (
          <>
          <div className="workspace-panels">
            <div className="card card--catalog">
              <div className="card-header">
                <div>
                  <h3>Module Library</h3>
                  <span className="card-subtitle">Browse and select a workflow to configure</span>
                </div>
                <span className="card-count">{filteredModules.length} modules</span>
              </div>

              <div className="card-body">
                <div className="filter-bar">
                  <input
                    className="field-input search-input"
                    placeholder="Search modules…"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />
                  <select className="field-input filter-select" value={scopeFilter} onChange={(event) => setScopeFilter(event.target.value)}>
                    <option value="All">All scopes</option>
                    {scopes.filter((s) => s !== "All").map((scope) => (
                      <option key={scope} value={scope}>
                        {scope}
                      </option>
                    ))}
                  </select>
                  <select
                    className="field-input filter-select"
                    value={categoryFilter}
                    onChange={(event) => setCategoryFilter(event.target.value)}
                  >
                    <option value="All">All categories</option>
                    {categories.filter((c) => c !== "All").map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                  <select className="field-input filter-select" value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
                    <option value="All">All risks</option>
                    <option value="safe">Routine</option>
                    <option value="caution">Needs care</option>
                    <option value="danger">High impact</option>
                  </select>
                </div>

                <div className="catalog-tabs" role="tablist">
                  <button
                    className={`catalog-tab ${catalogTab === "featured" ? "catalog-tab--active" : ""}`}
                    onClick={() => setCatalogTab("featured")}
                    type="button"
                    role="tab"
                    aria-selected={catalogTab === "featured"}
                  >
                    Featured ({featuredModules.length})
                  </button>
                  <button
                    className={`catalog-tab ${catalogTab === "advanced" ? "catalog-tab--active" : ""}`}
                    onClick={() => setCatalogTab("advanced")}
                    type="button"
                    role="tab"
                    aria-selected={catalogTab === "advanced"}
                  >
                    Advanced ({advancedModules.length})
                  </button>
                </div>

                {catalogError ? <div className="alert alert--danger">{catalogError}</div> : null}

                {catalogTab === "featured" ? (
                  featuredModules.length === 0 ? (
                    <div className="empty-state">
                      <span className="empty-state__icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24">
                          <path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />
                        </svg>
                      </span>
                      <p>No featured modules match your filters. Try adjusting search or filters.</p>
                    </div>
                  ) : (
                    <div className="module-grid">
                      {featuredModules.map((module) => (
                        <button
                          key={module.id}
                          className={`module-tile ${activeModule?.id === module.id ? "module-tile--selected" : ""}`}
                          onClick={() => selectModule(module.id)}
                          type="button"
                        >
                          <div className="module-tile__top">
                            <span className="module-icon">
                              <ModuleIcon category={module.category} mode={module.mode} />
                            </span>
                            <span className={`risk-badge risk-badge--${module.risk}`}>{riskLabel(module.risk)}</span>
                          </div>
                          <span className="module-kicker">{module.category}</span>
                          <h4>{module.title}</h4>
                          <p>{module.subtitle}</p>
                          <div className="module-badges">
                            <span className="scope-badge">{module.scope}</span>
                            <span className="mode-badge">{module.mode === "structured" ? "Guided" : "Advanced"}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  )
                ) : advancedModules.length === 0 ? (
                  <div className="empty-state">
                    <span className="empty-state__icon" aria-hidden="true">
                      <svg viewBox="0 0 24 24">
                        <path d="M4.5 6.5h15v11h-15z" />
                        <path d="M7.5 10l2 2-2 2M11.5 14h4" />
                      </svg>
                    </span>
                    <p>No advanced scripts match your filters.</p>
                  </div>
                ) : (
                  <div className="advanced-list">
                    {advancedModules.map((module) => (
                      <button
                        key={module.id}
                        className={`advanced-row ${activeModule?.id === module.id ? "advanced-row--selected" : ""}`}
                        onClick={() => selectModule(module.id)}
                        type="button"
                      >
                        <span className="advanced-row__icon">
                          <ModuleIcon category={module.category} mode={module.mode} />
                        </span>
                        <span className="advanced-row__body">
                          <strong>{module.title}</strong>
                          <small>{module.summary}</small>
                        </span>
                        <span className={`risk-badge risk-badge--${module.risk}`}>{riskLabel(module.risk)}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="card card--details">
              <div className="card-header">
                <div>
                  <h3>Configure & Run</h3>
                  <span className="card-subtitle">{activeModule?.script_filename ?? "Select a module from the library"}</span>
                </div>
                {activeModule ? (
                  <span className={`status-pill status-pill--${activeModule.risk === "danger" ? "danger" : activeModule.risk === "caution" ? "active" : "success"}`}>
                    {riskLabel(activeModule.risk)}
                  </span>
                ) : null}
              </div>

              <div className="card-body">
                {activeModule ? (
                  <>
                    <div className="script-summary">
                      <span className="module-icon module-icon--large">
                        <ModuleIcon category={activeModule.category} mode={activeModule.mode} />
                      </span>
                      <div className="script-summary__title">
                        <div className="meta-row">
                          <span className="scope-badge">{activeModule.scope}</span>
                          <span className="scope-badge">{activeModule.category}</span>
                        </div>
                        <h4>{activeModule.title}</h4>
                        <p>{activeModule.summary}</p>
                      </div>
                    </div>

                    {banner ? <div className={`alert alert--${banner.tone}`}>{banner.message}</div> : null}
                    {activeModule.caution_message ? <div className="alert alert--warning">{activeModule.caution_message}</div> : null}

                    {activeModule.indications.length > 0 ? (
                      <div className="indication-list">
                        {activeModule.indications.map((item) => (
                          <div key={item} className="indication-item">
                            {item}
                          </div>
                        ))}
                      </div>
                    ) : null}

                    {activeModuleInputFields.length > 0 ? (
                      <section className="detail-section">
                        <div className="detail-section__header">
                          <h4>Configuration</h4>
                        </div>
                        <div className="form-grid">
                          {activeModuleInputFields.map((field) => renderField(activeModule, field))}
                        </div>
                      </section>
                    ) : null}

                    {activeModuleToggleFields.length > 0 ? (
                      <section className="detail-section">
                        <div className="detail-section__header">
                          <h4>Options</h4>
                        </div>
                        <div className="toggle-grid">
                          {activeModuleToggleFields.map((field) => renderField(activeModule, field))}
                        </div>
                      </section>
                    ) : null}

                    <div className="command-preview">
                      <span>Launch command</span>
                      <code>python {activeModule.script_filename}</code>
                    </div>

                    {activeModule.usage_examples.length > 0 ? (
                      <section className="detail-section">
                        <div className="detail-section__header">
                          <h4>CLI examples</h4>
                        </div>
                        <div className="example-list">
                          {activeModule.usage_examples.map((example) => (
                            <div key={example} className="example-chip">
                              {example}
                            </div>
                          ))}
                        </div>
                      </section>
                    ) : null}

                    <div className="path-chip">{activeModule.script_path}</div>

                    <div className="action-row">
                      <button className="primary-button" onClick={() => void launchModule()} type="button">
                        {activeModule.run_label}
                      </button>
                      {activeJob?.status === "pending_approval" ? (
                        <>
                          <button className="primary-button" onClick={() => void approveJob(activeJob.id)} type="button">
                            Approve job
                          </button>
                          <button className="danger-button" onClick={() => void rejectJob(activeJob.id)} type="button">
                            Reject
                          </button>
                        </>
                      ) : null}
                      {activeJob?.status === "running" || activeJob?.status === "terminating" ? (
                        <button className="danger-button" onClick={() => void terminateJob(activeJob.id)} type="button">
                          Stop job
                        </button>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <span className="empty-state__icon" aria-hidden="true">
                      <svg viewBox="0 0 24 24">
                        <path d="M12 5v14M5 12h14" />
                      </svg>
                    </span>
                    <p>Select a module from the library to view its configuration and launch options.</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="card card--console">
            <div className="card-header">
              <div>
                <h3>Live Console</h3>
                <span className="card-subtitle">{activeJob ? activeJob.script_filename : "No job selected"}</span>
              </div>
              {activeJob ? (
                <span className={`status-pill status-pill--${statusTone(activeJob.status)}`}>{activeJob.status}</span>
              ) : null}
            </div>

            <div className="card-body">
              {streamError ? <div className="alert alert--warning">{streamError}</div> : null}
              {activeJob?.status === "pending_approval" ? (
                <div className="alert alert--warning">
                  This job requires approval before the worker pool will execute it.
                  <div className="action-row" style={{ marginTop: 12 }}>
                    <button className="primary-button" onClick={() => void approveJob(activeJob.id)} type="button">
                      Approve and run
                    </button>
                    <button className="danger-button" onClick={() => void rejectJob(activeJob.id)} type="button">
                      Reject job
                    </button>
                  </div>
                </div>
              ) : null}

              {activeJob ? (
                <>
                  <div className="job-meta-bar">
                    <div className="job-meta-bar__item">
                      <label>Job ID</label>
                      <code>{activeJob.id.slice(0, 8)}…</code>
                    </div>
                    <div className="job-meta-bar__item">
                      <label>Started</label>
                      <span>{formatTimestamp(activeJob.started_at ?? activeJob.created_at)}</span>
                    </div>
                    <div className="job-meta-bar__item">
                      <label>Exit code</label>
                      <span>{activeJob.exit_code ?? "—"}</span>
                    </div>
                    <div className="job-meta-bar__item">
                      <label>Module</label>
                      <span>{moduleForJob(activeJob)?.title ?? activeJob.script_filename}</span>
                    </div>
                    <div className="job-meta-bar__item">
                      <label>Args</label>
                      <span>{activeJob.raw_args || "none"}</span>
                    </div>
                  </div>

                  <div className="console-shell">
                    {activeJob.logs.length === 0 ? (
                      <span className="console-waiting">Waiting for output…</span>
                    ) : (
                      activeJob.logs.map((line, index) => (
                        <div key={`${activeJob.id}-${index}`} className="console-line">
                          {line || " "}
                        </div>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <div className="empty-state" style={{ padding: "32px 16px" }}>
                  <span className="empty-state__icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24">
                      <path d="M4.5 6.5h15v11h-15z" />
                      <path d="M7.5 10l2 2-2 2M11.5 14h4" />
                    </svg>
                  </span>
                  <p>Run a module or select a recent job from the sidebar to view live logs.</p>
                </div>
              )}
            </div>
          </div>
          </>
          ) : null}
        </div>
      </main>
    </div>
  );
}
