alter table query_history
    add column if not exists conversation_id text;

create index if not exists idx_query_history_conversation
    on query_history (conversation_id, created_at desc);
