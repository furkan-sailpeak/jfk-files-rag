import { createClient } from '@supabase/supabase-js';

// Both values are safe to ship in the bundle: the anon key is a public
// client key, and every table it can reach is protected by Row Level
// Security. The service_role key must NEVER appear in frontend code.
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

// TEMPORARILY DISABLED (see todo.md): the sign-in UI and the quota gate are
// switched off for now. Set VITE_AUTH_ENABLED=true to bring them back — the
// Auth component and all call sites are still wired up.
const AUTH_FLAG = import.meta.env.VITE_AUTH_ENABLED === 'true';

export const authEnabled =
  AUTH_FLAG && Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

if (AUTH_FLAG && !authEnabled) {
  console.warn(
    '[auth] VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY not set — running without authentication.'
  );
}

export const supabase = authEnabled
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    })
  : null;

/** Current access token, or null when signed out. Sent as a Bearer token so
 *  the backend can apply the signed-in quota instead of the trial limit. */
export async function getAccessToken() {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data?.session?.access_token ?? null;
}
