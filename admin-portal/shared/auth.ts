import { authConfig } from "./auth-config";

export type ArchivynUser = {
  id: string;
  username: string;
  name?: string;
  role?: string;
};

export type ArchivynAuthContext = {
  role?: string;
  roles?: string[];
};

export type ArchivynSession = {
  user: ArchivynUser;
  authContext: ArchivynAuthContext;
};

type MeResponse = {
  user: ArchivynUser;
  authContext?: ArchivynAuthContext;
};

type ErrorResponse = {
  error?: string;
  message?: string;
};

function authUrl(path: string): string {
  const base = authConfig.adminApiUrl;
  if (!base) {
    throw new Error("NEXT_PUBLIC_ADMIN_API_URL is not configured.");
  }
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as ErrorResponse;
    return payload.error ?? payload.message ?? `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export async function fetchSession(): Promise<ArchivynSession | null> {
  const response = await fetch(authUrl("/auth/me"), {
    method: "GET",
    credentials: "include",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (response.status === 401) {
    return null;
  }

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const payload = (await response.json()) as MeResponse;
  return {
    user: payload.user,
    authContext: payload.authContext ?? {},
  };
}

export async function loginWithPassword(username: string, password: string): Promise<ArchivynSession> {
  const response = await fetch(authUrl("/auth/login"), {
    method: "POST",
    credentials: "include",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const session = await fetchSession();
  if (!session) {
    throw new Error("Login succeeded but session could not be verified.");
  }
  return session;
}

export async function logoutSession(): Promise<void> {
  try {
    await fetch(authUrl("/auth/logout"), {
      method: "POST",
      credentials: "include",
      headers: { Accept: "application/json" },
    });
  } catch {
    // Best-effort; admin API may not expose logout yet.
  }
}
