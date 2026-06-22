export type ArchivynRole = "company_admin" | "school_admin" | "teacher" | string;

function envFlag(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") {
    return defaultValue;
  }
  return raw === "1" || raw.toLowerCase() === "true" || raw.toLowerCase() === "yes";
}

function parseRoles(raw: string | undefined, fallback: ArchivynRole[]): ArchivynRole[] {
  if (!raw?.trim()) {
    return fallback;
  }
  return raw
    .split(",")
    .map((role) => role.trim())
    .filter(Boolean);
}

export const authConfig = {
  adminApiUrl: (process.env.NEXT_PUBLIC_ADMIN_API_URL ?? "").replace(/\/$/, ""),
  required: envFlag("NEXT_PUBLIC_AUTH_REQUIRED", true),
  allowedRoles: parseRoles(process.env.NEXT_PUBLIC_AUTH_ALLOWED_ROLES, ["company_admin"]),
  cookieName: "archivyn_session",
};

export function isAuthConfigured(): boolean {
  return Boolean(authConfig.adminApiUrl);
}

export function isAuthMisconfigured(): boolean {
  return authConfig.required && !isAuthConfigured();
}

export function roleIsAllowed(role: string | undefined, roles: string[] | undefined): boolean {
  const allowed = new Set(authConfig.allowedRoles);
  if (role && allowed.has(role)) {
    return true;
  }
  return (roles ?? []).some((entry) => allowed.has(entry));
}
