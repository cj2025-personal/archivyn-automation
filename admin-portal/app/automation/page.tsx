import { Suspense } from "react";

import { AuthGate } from "../components/auth-gate";
import { AutomationModule } from "./script-runner-module";

export default function AutomationPage() {
  return (
    <Suspense
      fallback={
        <div className="auth-screen">
          <div className="auth-card auth-card--centered">
            <h1>Loading automation console…</h1>
          </div>
        </div>
      }
    >
      <AuthGate>
        <AutomationModule />
      </AuthGate>
    </Suspense>
  );
}
