create table if not exists public.query_answers (
    id bigserial primary key,
    query_hash text not null,
    question text not null,
    normalized_question text not null,
    answer text,
    sources jsonb,
    namespace text not null default 'epstein-docs',
    cached boolean default false,
    error text,
    ask_count integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    last_asked_at timestamptz not null default now(),
    unique (namespace, query_hash)
);

alter table public.query_history
add column if not exists user_id uuid references auth.users(id) on delete cascade,
add column if not exists query_answer_id bigint references public.query_answers(id) on delete cascade,
add column if not exists query_hash text,
add column if not exists ask_count integer not null default 1,
add column if not exists last_asked_at timestamptz not null default now();

create index if not exists idx_query_history_user_conversation_created
on public.query_history (user_id, conversation_id, created_at desc);

create unique index if not exists idx_query_history_user_conversation_hash
on public.query_history (user_id, coalesce(conversation_id, ''), query_hash)
where user_id is not null and query_hash is not null;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_query_answers_updated_at on public.query_answers;

create trigger set_query_answers_updated_at
before update on public.query_answers
for each row
execute function public.set_updated_at();
