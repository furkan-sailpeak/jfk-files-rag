import React, { useState } from 'react';
import { supabase } from './supabaseClient';
import { COUNTRIES } from './countries';

/**
 * Magic-link auth modal.
 *
 * Two modes share one Supabase call (signInWithOtp):
 *  - "signup" collects country + institution, passed as options.data. Supabase
 *    stores that on auth.users.raw_user_meta_data, and the on_auth_user_created
 *    trigger copies it into public.profiles.
 *  - "signin" sets shouldCreateUser:false, so an unknown address is rejected
 *    rather than silently creating a profile with no country/institution.
 */
export default function Auth({ open, onClose, reason }) {
  const [mode, setMode] = useState('signup');
  const [email, setEmail] = useState('');
  const [country, setCountry] = useState('');
  const [institution, setInstitution] = useState('');
  const [status, setStatus] = useState('idle'); // idle | sending | sent | error
  const [error, setError] = useState('');

  if (!open) return null;

  const isSignup = mode === 'signup';

  const submit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email.trim()) {
      setError('Please enter your email address.');
      return;
    }
    if (isSignup && (!country || !institution.trim())) {
      setError('Country and institution are required to create an account.');
      return;
    }

    setStatus('sending');
    try {
      const { error: authError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: {
          shouldCreateUser: isSignup,
          emailRedirectTo: window.location.origin,
          data: isSignup
            ? { country, institution: institution.trim() }
            : undefined,
        },
      });
      if (authError) throw authError;
      setStatus('sent');
    } catch (err) {
      setStatus('error');
      const msg = err?.message || 'Could not send the sign-in link.';
      setError(
        /signups not allowed|not found/i.test(msg)
          ? 'No account found for that address. Switch to "Create account" to register.'
          : msg
      );
    }
  };

  return (
    <div className="auth-overlay" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <div className="auth-modal">
        <button className="auth-close" onClick={onClose} aria-label="Close">×</button>

        {status === 'sent' ? (
          <div className="auth-sent">
            <h2 id="auth-title">Check your inbox</h2>
            <p>
              We sent a one-time sign-in link to <strong>{email}</strong>.
              Open it on this device to continue. The link expires in one hour.
            </p>
            <p className="auth-hint">
              Nothing arrived? Check your spam folder, or{' '}
              <button className="auth-link" onClick={() => setStatus('idle')}>
                try a different address
              </button>.
            </p>
          </div>
        ) : (
          <>
            <h2 id="auth-title">{isSignup ? 'Create an account' : 'Sign in'}</h2>

            {reason && <p className="auth-reason">{reason}</p>}

            <p className="auth-blurb">
              This is a research tool built for a KU Leuven master's thesis.
              We ask for your country and institution to report how the archive
              is used. No password required — we email you a sign-in link.
            </p>

            <form onSubmit={submit}>
              <label htmlFor="auth-email">Email address</label>
              <input
                id="auth-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@university.edu"
                required
              />

              {isSignup && (
                <>
                  <label htmlFor="auth-country">Country</label>
                  <select
                    id="auth-country"
                    value={country}
                    onChange={(e) => setCountry(e.target.value)}
                    required
                  >
                    <option value="">Select a country…</option>
                    {COUNTRIES.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>

                  <label htmlFor="auth-institution">Institution or organisation</label>
                  <input
                    id="auth-institution"
                    type="text"
                    value={institution}
                    onChange={(e) => setInstitution(e.target.value)}
                    placeholder="University, newsroom, or “Independent”"
                    maxLength={120}
                    required
                  />
                </>
              )}

              {error && <p className="auth-error">{error}</p>}

              <button className="auth-submit" type="submit" disabled={status === 'sending'}>
                {status === 'sending' ? 'Sending link…' : 'Email me a sign-in link'}
              </button>
            </form>

            <p className="auth-switch">
              {isSignup ? 'Already registered?' : 'First time here?'}{' '}
              <button
                className="auth-link"
                onClick={() => { setMode(isSignup ? 'signin' : 'signup'); setError(''); }}
              >
                {isSignup ? 'Sign in' : 'Create an account'}
              </button>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
