-- The two shared tables R3 reads but does not create: the job sites and the
-- job postings found on them. Only the columns the portal uses are here.
-- Everything an account owns (r3_users, r3_user_jobs, r3_resumes, labels,
-- settings, events, ...) R3 creates on demand at startup.

CREATE TABLE public.job_sites (
    url text PRIMARY KEY,
    name text NOT NULL
);

CREATE TABLE public.scraped_jobs (
    id bigserial PRIMARY KEY,
    site_url text NOT NULL REFERENCES public.job_sites (url),
    job_url text NOT NULL UNIQUE,
    apply_url text,
    position text NOT NULL,
    company text,
    location text,
    posted_date date,
    scraped_at timestamp with time zone DEFAULT now() NOT NULL,
    role_category text,
    employment text,
    salary text,
    salary_min integer,
    salary_max integer,
    salary_period text,
    tech_tags text[] DEFAULT '{}'::text[] NOT NULL,
    location_kind text,
    description text,
    -- Whether the posting is still up: 'alive', 'expired' (gone) or 'unknown'.
    last_validation text CHECK (last_validation IS NULL
                                OR last_validation IN ('alive', 'expired', 'unknown', 'skipped')),
    last_validated_at timestamp with time zone,
    verification_status text CHECK (verification_status IS NULL
                                    OR verification_status IN ('verified', 'corrected', 'conflict',
                                                               'mismatched_job', 'failed'))
);

CREATE INDEX scraped_jobs_site_scraped_idx ON public.scraped_jobs (site_url, scraped_at);
CREATE INDEX scraped_jobs_role_category_idx ON public.scraped_jobs (role_category);
CREATE INDEX scraped_jobs_location_kind_idx ON public.scraped_jobs (location_kind);
CREATE INDEX scraped_jobs_tech_tags_gin ON public.scraped_jobs USING gin (tech_tags);
