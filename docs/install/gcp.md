# GCP install — Cloud Run + Cloud SQL + GCE Neo4j

> **Validated 2026-04-28** — automated deploy reached `smoke rc=0` against
> `https://attestor-api-743444662382.us-central1.run.app`
> (Cloud Run + `db-f1-micro` Cloud SQL + `e2-small` GCE Neo4j VM).
> Total provisioning + smoke ~17 min wall-clock, ~$22/mo.

The canonical Attestor deploy on GCP:

```
        ┌──────────────────┐         ┌────────────────────┐
        │  Cloud Run       │         │  GCE e2-small VM   │
client →│  attestor-api    │ ──TCP──→│  neo4j:5.24        │
        └──────────────────┘   ↑     │  (bolt:// 7687)    │
                  │            │     └────────────────────┘
                  │            │     ┌────────────────────┐
                  │            └────→│  Cloud SQL PG 16   │
                  │                  │  + pgvector        │
                  ▼                  │  (private IP)      │
        ┌──────────────────┐         └────────────────────┘
        │ Secret Manager   │
        │ (5 secrets)      │
        └──────────────────┘
```

Estimated monthly cost (cheapest tiers): ~**$20/month**
(Cloud SQL `db-f1-micro` ~$7, GCE `e2-small` ~$13, Cloud Run within free tier).

## 0. Prereqs

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT>
gcloud config set run/region us-central1
gcloud config set compute/region us-central1

gcloud services enable \
  run.googleapis.com sqladmin.googleapis.com compute.googleapis.com \
  secretmanager.googleapis.com servicenetworking.googleapis.com \
  vpcaccess.googleapis.com
```

Native registry for GCP pulls: `us-central1-docker.pkg.dev/coral-marker-452616-n4/attestor/attestor:api-4.0.0a5`

## 1. Secrets

```bash
PROJECT=$(gcloud config get-value project)
PG_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
NEO4J_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)

for name in postgres-password neo4j-password voyage-api-key openai-api-key openrouter-api-key; do
  gcloud secrets create $name --replication-policy=automatic 2>/dev/null || true
done

printf "%s" "$PG_PASSWORD"         | gcloud secrets versions add postgres-password   --data-file=-
printf "%s" "$NEO4J_PASSWORD"      | gcloud secrets versions add neo4j-password      --data-file=-
printf "%s" "$VOYAGE_API_KEY"      | gcloud secrets versions add voyage-api-key      --data-file=-
printf "%s" "$OPENAI_API_KEY"      | gcloud secrets versions add openai-api-key      --data-file=-
printf "%s" "$OPENROUTER_API_KEY"  | gcloud secrets versions add openrouter-api-key  --data-file=-
```

## 2. Cloud SQL Postgres 16 + pgvector

```bash
gcloud sql instances create attestor-pg \
  --database-version=POSTGRES_16 \
  --edition=ENTERPRISE \
  --tier=db-f1-micro \
  --region=us-central1 \
  --storage-size=10GB \
  --storage-type=HDD \
  --no-backup \
  --root-password="$PG_PASSWORD"

gcloud sql databases create attestor --instance=attestor-pg

# pgvector extension (run once after the instance is up)
gcloud sql connect attestor-pg --user=postgres --database=attestor <<EOF
CREATE EXTENSION IF NOT EXISTS vector;
EOF
```

> Cloud SQL takes 5–10 minutes to come up. `gcloud sql operations list --instance attestor-pg` to monitor.

> **`--edition=ENTERPRISE` is required** — without it Cloud SQL defaults
> new instances to `ENTERPRISE_PLUS`, which rejects `db-f1-micro` with
> *"Invalid Tier for ENTERPRISE_PLUS Edition"*.

> **There is NO `cloudsql.enable_pgvector` startup flag.** Don't try to
> set it via `--database-flags`. pgvector is enabled per-database with
> `CREATE EXTENSION vector;` (above) once the instance is up.

> **Reaching Cloud SQL from Cloud Run** — the Serverless VPC connector
> doesn't NAT through a stable public IP, so a "public-IP + authorized
> networks" Cloud SQL setup needs `authorized-networks=0.0.0.0/0` (with
> SSL required) for the demo. For production, use Cloud SQL **Private
> IP** via Private Services Access — same VPC as the connector.

## 3. GCE Neo4j VM

Cloud Run is HTTP-only; Neo4j needs `bolt://` (TCP/7687), so we run it on
a tiny Compute Engine VM in the same region.

```bash
gcloud compute instances create-with-container attestor-neo4j \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --container-image=neo4j:5.24-community \
  --container-env="NEO4J_AUTH=neo4j/$NEO4J_PASSWORD" \
  --container-env="NEO4J_PLUGINS=[\"graph-data-science\"]" \
  --tags=neo4j

gcloud compute firewall-rules create allow-neo4j-bolt \
  --target-tags=neo4j \
  --source-ranges=10.8.0.0/28 \
  --allow=tcp:7687
```

> The source-range `10.8.0.0/28` is the Serverless VPC Connector subnet
> we'll create next; only Cloud Run will be allowed to reach Neo4j.

> **Neo4j first-boot is ~2 min on e2-small** (image pull + JVM warmup +
> pagecache init). Cloud Run readiness probes will see `NEO4J_NOT_READY`
> for that window. Either set min-instances=1 + a generous startup probe
> grace period, or wait 2 min before the first smoke run.

## 4. VPC connector for Cloud Run → Cloud SQL + Neo4j

```bash
gcloud compute networks vpc-access connectors create attestor-conn \
  --region=us-central1 \
  --network=default \
  --range=10.8.0.0/28
```

## 5. Cloud Run service

```bash
NEO4J_IP=$(gcloud compute instances describe attestor-neo4j --zone=us-central1-a \
  --format='value(networkInterfaces[0].networkIP)')
PG_IP=$(gcloud sql instances describe attestor-pg --format='value(ipAddresses[0].ipAddress)')

gcloud run deploy attestor-api \
  --image=us-central1-docker.pkg.dev/$PROJECT/attestor/attestor:api-4.0.0a5 \
  --region=us-central1 \
  --port=8080 \
  --min-instances=1 \
  --memory=512Mi --cpu=1 \
  --vpc-connector=attestor-conn --vpc-egress=private-ranges-only \
  --allow-unauthenticated \
  --set-env-vars=POSTGRES_URL=postgresql://${PG_IP}:5432,POSTGRES_DATABASE=attestor,POSTGRES_USERNAME=postgres,POSTGRES_SSLMODE=require,NEO4J_URI=bolt://${NEO4J_IP}:7687,NEO4J_USERNAME=neo4j,ATTESTOR_V4=1,ATTESTOR_DISABLE_LOCAL_EMBED=1 \
  --set-secrets=POSTGRES_PASSWORD=postgres-password:latest,NEO4J_PASSWORD=neo4j-password:latest,VOYAGE_API_KEY=voyage-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,OPENROUTER_API_KEY=openrouter-api-key:latest
```

## 6. Smoke

```bash
URL=$(gcloud run services describe attestor-api --region=us-central1 --format='value(status.url)')
.venv/bin/python scripts/smoke_api.py --url "$URL"
```

Expected: `rc=0` with all 4 health checks + 3 writes + 3 recalls passing.

## 7. Teardown

```bash
gcloud run services delete attestor-api --region=us-central1 -q
gcloud compute instances delete attestor-neo4j --zone=us-central1-a -q
gcloud compute firewall-rules delete allow-neo4j-bolt -q
gcloud compute networks vpc-access connectors delete attestor-conn --region=us-central1 -q
gcloud sql instances delete attestor-pg -q
for s in postgres-password neo4j-password voyage-api-key openai-api-key openrouter-api-key; do
  gcloud secrets delete $s -q
done
```
