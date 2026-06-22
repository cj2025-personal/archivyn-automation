"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { useAuth } from "@/app/providers/auth-provider";

export function AuthGate({ children }: { children: ReactNode }) {
  const { authEnabled, authMisconfigured, loading, isAuthenticated, isAuthorized, error } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (!authEnabled || loading || pathname === "/login") {
      return;
    }

    if (!isAuthenticated) {
      const query = new URLSearchParams();
      const returnUrl = `${pathname}${searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
      query.set("returnUrl", returnUrl);
      router.replace(`/login?${query.toString()}`);
    }
  }, [authEnabled, isAuthenticated, loading, pathname, router, searchParams]);

  if (authMisconfigured) {
    return (
      <div className="auth-screen">
        <div className="auth-card auth-card--centered">
          <p className="eyebrow">Configuration required</p>
          <h1>Admin API URL missing</h1>
          <p className="auth-copy">
            Set <code>NEXT_PUBLIC_ADMIN_API_URL</code> in Vercel (for example{" "}
            <code>https://&lt;your-admin-api&gt;/api</code>) so this portal can verify Archivyn sessions.
          </p>
        </div>
      </div>
    );
  }

  if (!authEnabled) {
    return <>{children}</>;
  }

  if (loading) {
    return (
      <div className="auth-screen">
        <div className="auth-card auth-card--centered">
          <p className="eyebrow">Archivyn Admin</p>
          <h1>Checking session…</h1>
          <p className="auth-copy">Verifying your login with the admin API.</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="auth-screen">
        <div className="auth-card auth-card--centered">
          <p className="eyebrow">Archivyn Admin</p>
          <h1>Redirecting to sign in</h1>
          <p className="auth-copy">Use the same username and password as the main admin portal.</p>
          <Link className="primary-button auth-submit" href="/login">
            Go to sign in
          </Link>
        </div>
      </div>
    );
  }

  if (!isAuthorized) {
    return (
      <div className="auth-screen">
        <div className="auth-card auth-card--centered">
          <p className="eyebrow">Access denied</p>
          <h1>Insufficient permissions</h1>
          <p className="auth-copy">
            Your account is signed in but does not have a role allowed for automation operations.
          </p>
          {error ? <p className="auth-error">{error}</p> : null}
          <Link className="secondary-button secondary-button--link" href="/login">
            Switch account
          </Link>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
