import { Suspense } from "react";

import LoginForm from "./login-form";

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="auth-screen">
          <div className="auth-card auth-card--centered">
            <h1>Loading sign in…</h1>
          </div>
        </div>
      }
    >
      <LoginForm />
    </Suspense>
  );
}
