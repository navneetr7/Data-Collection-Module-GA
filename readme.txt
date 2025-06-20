env example

#Replace with your secure API key 
API_KEY=

#Obtain from Google Cloud Console (https://cloud.google.com)
GEMINI_API_KEY=

#Obtain from Supabase Project Settings > API (Project URL and anon key)
SUPABASE_URL=
SUPABASE_KEY=

#Create a bucket in Supabase Storage (e.g., 'data-collection-files')
SUPABASE_STORAGE_BUCKET=data-collection-files

#Obtain from Zendesk Auth
ZENDESK_SUBDOMAIN=
ZENDESK_EMAIL=
ZENDESK_API_TOKEN=


#Create Slack Webhook
SLACK_WEBHOOK_URL=

#Obtain from Zendesk custom field (Admin Center > Tickets > Fields > Data Collection Link)
ZENDESK_FIELD_ID=

#Obtain from Zendesk custom field (Admin Center > Tickets > Fields > Customer Answers)
ZENDESK_CUSTOMER_ANSWERS_FIELD_ID=

#URL Backend
BASE_URL=

#URL Frontend
FRONTEND_BASE_URL=




SQL For Supbase table creation

-- Table for data collection forms
create table public.data_collection (
    uuid uuid primary key,
    ticket_id text not null,
    customer_email text not null,
    questions jsonb not null,
    status text not null check (status in ('pending', 'completed', 'partial')),
    total_upload_size bigint default 0,
    created_at timestamptz default now()
);

create index idx_data_collection_ticket_id on public.data_collection(ticket_id);

alter table public.data_collection enable row level security;

create policy "Service role full access"
on public.data_collection
for all
using (auth.role() = 'service_role');

create policy "Authenticated read"
on public.data_collection
for select
using (auth.role() = 'authenticated');

-- Storage bucket policies for 'data-collection-files'
create policy "Allow read access to files for authenticated users"
on storage.objects
for select
using (
  bucket_id = 'data-collection-files'
);

create policy "Allow service role full access to files"
on storage.objects
for all
using (
  auth.role() = 'service_role'
  and bucket_id = 'data-collection-files'
);



------------------


For file storage create bucket with name 
data-collection-files