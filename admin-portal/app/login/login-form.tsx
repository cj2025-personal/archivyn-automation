"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { useAuth } from "@/app/providers/auth-provider";

export default function LoginForm() {
  const { authEnabled, loading, isAuthenticated, isAuthorized, login } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const returnUrl = searchParams.get("returnUrl") || "/automation";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && (!authEnabled || (isAuthenticated && isAuthorized))) {
      router.replace(returnUrl);
    }
  }, [authEnabled, isAuthenticated, isAuthorized, loading, returnUrl, router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username.trim(), password);
      router.replace(returnUrl);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Sign in failed.");
    } finally {
      setSubmitting(false);
    }
  }

  if (!authEnabled) {
    return (
      <div className="auth-screen">
        <div className="auth-card auth-card--centered">
          <p className="eyebrow">Automation Portal</p>
          <h1>Authentication disabled</h1>
          <p className="auth-copy">Set NEXT_PUBLIC_ADMIN_API_URL and NEXT_PUBLIC_AUTH_REQUIRED to enable Archivyn login.</p>
          <Link className="primary-button auth-submit" href="/automation">
            Continue to automation
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-card__header">
          <p className="eyebrow">Archivyn Admin</p>
          <h1>Sign in to Automation</h1>
          <p className="auth-copy">
            Use the same username and password as the main admin portal. Your session is stored as a secure cookie on
            the admin API domain.
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="form-field">
            <span className="field-title">Username</span>
            <input
              className="field-input"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>

          <label className="form-field">
            <span className="field-title">Password</span>
            <input
              className="field-input"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>

          {error ? <p className="auth-error">{error}</p> : null}

          <button className="primary-button auth-submit" type="submit" disabled={submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <p className="auth-footnote">
          Already signed in on Archivyn Admin? This page will reuse that session when your browser sends the admin API
          cookie.
        </p>
      </div>
    </div>
  );
}
