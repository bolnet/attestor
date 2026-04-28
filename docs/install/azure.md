# Azure install — Container Apps + Postgres Flex + Neo4j Container App

> **Validated 2026-04-28** — automated deploy reached `smoke rc=0` against
> `https://memwright.wittyisland-00214c25.eastus.azurecontainerapps.io`
> (Container Apps env + `Standard_B1ms` Postgres Flex + Neo4j as a
> separate Container App with internal TCP ingress on :7687).
> Total provisioning + smoke ~37 min wall-clock (fought several Azure-
> specific gotchas — see end of doc), ~$95/mo at the 2vCPU/4GiB tier
> the agent ended up using; ~$25/mo achievable at smaller sizes.

The canonical Attestor deploy on Azure:

```
        ┌──────────────────────────────┐
        │  Container Apps environment  │
        │                              │
client →│  ┌─────────────────────────┐ │
        │  │ attestor-api  (HTTPS)   │ │
        │  └────────┬────────────────┘ │
        │           │internal           │
        │  ┌────────▼────────────────┐ │
        │  │ attestor-neo4j (bolt)   │ │
        │  └─────────────────────────┘ │
        └──────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Postgres Flexible Server │
        │   + pgvector             │
        └──────────────────────────┘
```

Estimated monthly cost (cheapest tiers): ~**$25/month**
(Postgres Flex `Standard_B1ms` ~$13, Container Apps consumption-only ~$10).

## 0. Prereqs

```bash
az login
az account set --subscription <YOUR_SUBSCRIPTION>

az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.DBforPostgreSQL
```

Native registry for Azure pulls: `memwright.azurecr.io/attestor:api-4.0.0a5`
(private — see step 1 to import the image into your own ACR).

## 1. Resource group + import the image into your ACR

```bash
RG=attestor-bench
LOC=eastus

az group create -n $RG -l $LOC

# Use an existing ACR, or create a new one:
ACR=attestor$RANDOM
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true

# Import the public image into your private ACR (no rebuild needed):
az acr import --name $ACR \
  --source public.ecr.aws/m6h5j7o3/attestor:api-4.0.0a5 \
  --image attestor:api-4.0.0a5
```

## 2. Postgres Flex + pgvector

```bash
PG_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)

az postgres flexible-server create \
  --resource-group $RG \
  --name attestor-pg \
  --location $LOC \
  --tier Burstable --sku-name Standard_B1ms \
  --version 16 \
  --storage-size 32 \
  --admin-user postgres \
  --admin-password "$PG_PASSWORD" \
  --public-access 0.0.0.0 \
  --yes

az postgres flexible-server db create \
  --resource-group $RG --server-name attestor-pg --database-name attestor

# Enable pgvector
az postgres flexible-server parameter set \
  --resource-group $RG --server-name attestor-pg \
  --name azure.extensions --value VECTOR

az postgres flexible-server execute \
  --name attestor-pg --admin-user postgres --admin-password "$PG_PASSWORD" \
  --database-name attestor \
  --querytext "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 3. Container Apps environment + Neo4j

```bash
ENV_NAME=attestor-env
az containerapp env create -g $RG -n $ENV_NAME -l $LOC

NEO4J_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)

# Neo4j as a separate Container App with INTERNAL-only ingress on TCP 7687
az containerapp create \
  --resource-group $RG --name attestor-neo4j --environment $ENV_NAME \
  --image neo4j:5.24-community \
  --target-port 7687 --transport tcp --ingress internal \
  --min-replicas 1 --max-replicas 1 \
  --cpu 0.5 --memory 1.0Gi \
  --env-vars NEO4J_AUTH=neo4j/$NEO4J_PASSWORD NEO4J_PLUGINS='["graph-data-science"]'
```

## 4. The API Container App

```bash
PG_HOST=$(az postgres flexible-server show -g $RG -n attestor-pg --query fullyQualifiedDomainName -o tsv)
NEO4J_FQDN=$(az containerapp show -g $RG -n attestor-neo4j --query properties.configuration.ingress.fqdn -o tsv)
ACR_LOGIN=$(az acr show -g $RG -n $ACR --query loginServer -o tsv)
ACR_USER=$(az acr credential show -g $RG -n $ACR --query username -o tsv)
ACR_PASS=$(az acr credential show -g $RG -n $ACR --query 'passwords[0].value' -o tsv)

az containerapp create \
  --resource-group $RG --name attestor-api --environment $ENV_NAME \
  --image $ACR_LOGIN/attestor:api-4.0.0a5 \
  --registry-server $ACR_LOGIN --registry-username $ACR_USER --registry-password "$ACR_PASS" \
  --target-port 8080 --ingress external \
  --min-replicas 1 --max-replicas 3 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets postgres-password="$PG_PASSWORD" neo4j-password="$NEO4J_PASSWORD" voyage-api-key="$VOYAGE_API_KEY" openai-api-key="$OPENAI_API_KEY" openrouter-api-key="$OPENROUTER_API_KEY" \
  --env-vars \
    POSTGRES_URL=postgresql://$PG_HOST:5432 \
    POSTGRES_DATABASE=attestor \
    POSTGRES_USERNAME=postgres \
    POSTGRES_SSLMODE=require \
    POSTGRES_PASSWORD=secretref:postgres-password \
    NEO4J_URI=bolt://$NEO4J_FQDN:7687 \
    NEO4J_USERNAME=neo4j \
    NEO4J_PASSWORD=secretref:neo4j-password \
    ATTESTOR_V4=1 \
    ATTESTOR_DISABLE_LOCAL_EMBED=1 \
    VOYAGE_API_KEY=secretref:voyage-api-key \
    OPENAI_API_KEY=secretref:openai-api-key \
    OPENROUTER_API_KEY=secretref:openrouter-api-key
```

## 5. Smoke

```bash
URL=https://$(az containerapp show -g $RG -n attestor-api --query properties.configuration.ingress.fqdn -o tsv)
.venv/bin/python scripts/smoke_api.py --url "$URL"
```

## 6. Teardown

```bash
az group delete --name attestor-bench --yes --no-wait
```

(Single command nukes everything — secrets, Postgres, ACR, Container Apps environment, the lot.)

## Azure-specific gotchas

These cost the validation run a lot of time. Read them before you start.

1. **`btree_gist` is not allow-listed by default** on Postgres Flexible
   Server. Attestor's v4 schema uses it for exclusion constraints. Add
   it to the extension allow-list:
   ```bash
   az postgres flexible-server parameter set -g $RG -s attestor-pg \
     --name azure.extensions --value VECTOR,BTREE_GIST
   az postgres flexible-server restart -g $RG -n attestor-pg
   ```
2. **Empty/short PG passwords sometimes fail to propagate** even after
   `update --admin-password` reports success. If `psql` still errors
   with `password authentication failed` after a reset, do it once more
   with a different (longer) password — the second update tends to stick.
3. **Container Apps `secretref:` interpolation only works as the
   *entire* env value.** You can't embed a secretref inside a longer
   string (e.g., a connection URL). Use separate env vars:
   `POSTGRES_URL=postgresql://<host>:<port>` (no creds) plus
   `POSTGRES_USERNAME=<user>` plus `POSTGRES_PASSWORD=secretref:postgres-password`.
4. **HTTP startup probe on `/health` will time out**, even at 10s, because
   the api lazily initializes Postgres + Neo4j + Voyage on the first
   request. Switch both startupProbe and livenessProbe to `tcpSocket`
   on port 8080. Container reports healthy as soon as uvicorn binds;
   the heavy init runs on the first /health request.
5. **Use the SHORT Neo4j hostname** for the api's `NEO4J_URI`, not the
   full internal FQDN. The long form (`<app>.internal.<env-suffix>.<region>.azurecontainerapps.io`)
   resolves but TCP connect times out from the api Container App. The
   short form (`bolt://attestor-neo4j:7687`) connects on the first try.
