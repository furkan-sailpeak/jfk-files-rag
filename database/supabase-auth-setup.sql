-- ---------------------------------------------------------------------------
-- Supabase Auth setup for the JFK Files Research System
--
-- Run once in the Supabase SQL Editor (Dashboard > SQL Editor > New query).
-- Creates the profile table that captures country + institution at signup,
-- and the trigger that populates it from the magic-link signup metadata.
-- ---------------------------------------------------------------------------

-- 1. Profiles -----------------------------------------------------------------
create table if not exists public.profiles (
    id          uuid primary key references auth.users on delete cascade,
    email       text,
    country     text,
    institution text,
    created_at  timestamptz not null default now()
);

comment on table public.profiles is
    'Signup metadata for research-use reporting. One row per authenticated user.';

-- Row Level Security: a user may read and update only their own row.
-- Without RLS enabled, the anon API key could read every row in this table.
alter table public.profiles enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
    for select using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
    for update using (auth.uid() = id);

-- Deliberately no INSERT policy: rows are created solely by the trigger
-- below, which runs as security definer. Clients must not insert directly.

-- 2. Populate profile on signup -----------------------------------------------
-- The frontend passes country/institution through signInWithOtp's
-- options.data, which Supabase stores on auth.users.raw_user_meta_data.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.profiles (id, email, country, institution)
    values (
        new.id,
        new.email,
        nullif(trim(new.raw_user_meta_data ->> 'country'), ''),
        nullif(trim(new.raw_user_meta_data ->> 'institution'), '')
    )
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- 3. Rate limiting -------------------------------------------------------------
-- Mirrors what the Flask app bootstraps at startup; included so the schema is
-- reproducible from this file alone.
create table if not exists public.rate_limits (
    bucket       text primary key,
    count        integer not null default 0,
    window_start timestamptz not null default now()
);

create index if not exists idx_rate_limits_window
    on public.rate_limits (window_start);

-- The app writes here with the service-side DATABASE_URL connection, which
-- bypasses RLS. Enable RLS with no policies so the public anon key cannot
-- read or tamper with the counters.
alter table public.rate_limits enable row level security;

-- 4. Housekeeping --------------------------------------------------------------
-- Bucket keys embed their own time window, so old rows are inert rather than
-- wrong -- but they accumulate. Run periodically (Supabase cron, or manually).
--   delete from public.rate_limits where window_start < now() - interval '7 days';

-- 5. Reporting helper ----------------------------------------------------------
-- Signup breakdown for the thesis / KU Leuven reporting.
--   select country, institution, count(*)
--   from public.profiles
--   group by country, institution
--   order by count(*) desc;
