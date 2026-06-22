"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { authConfig, isAuthConfigured, isAuthMisconfigured, roleIsAllowed } from "@/shared/auth-config";
import {
  fetchSession,
  loginWithPassword,
  logoutSession,
  type ArchivynSession,
} from "@/shared/auth";

type AuthState = {
  session: ArchivynSession | null;
  loading: boolean;
  error: string | null;
  authEnabled: boolean;
  authMisconfigured: boolean;
  isAuthenticated: boolean;
  isAuthorized: boolean;
  refresh: () => Promise<void>;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

function sessionIsAuthorized(session: ArchivynSession | null): boolean {
  if (!session) {
    return false;
  }
  const role = session.authContext.role ?? session.user.role;
  return roleIsAllowed(role, session.authContext.roles);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const authMisconfigured = isAuthMisconfigured();
  const authEnabled = authConfig.required && isAuthConfigured();
  const [session, setSession] = useState<ArchivynSession | null>(null);
  const [loading, setLoading] = useState(authEnabled);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!authEnabled) {
      setSession(null);
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const nextSession = await fetchSession();
      setSession(nextSession);
    } catch (refreshError) {
      setSession(null);
      setError(refreshError instanceof Error ? refreshError.message : "Unable to verify session.");
    } finally {
      setLoading(false);
    }
  }, [authEnabled]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const login = useCallback(
    async (username: string, password: string) => {
      setError(null);
      const nextSession = await loginWithPassword(username, password);
      if (!sessionIsAuthorized(nextSession)) {
        await logoutSession();
        setSession(null);
        throw new Error("Your account does not have access to the automation portal.");
      }
      setSession(nextSession);
    },
    [],
  );

  const logout = useCallback(async () => {
    await logoutSession();
    setSession(null);
    setError(null);
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      session,
      loading,
      error,
      authEnabled,
      authMisconfigured,
      isAuthenticated: Boolean(session),
      isAuthorized: !authEnabled || sessionIsAuthorized(session),
      refresh,
      login,
      logout,
    }),
    [authEnabled, authMisconfigured, error, loading, login, logout, refresh, session],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider.");
  }
  return context;
}
