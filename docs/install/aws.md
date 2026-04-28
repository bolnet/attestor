# AWS install — App Runner + RDS Postgres + EC2 Neo4j

> **Validated 2026-04-28** — automated deploy reached `smoke rc=0` against
> `https://r3cmjmnuw3.us-east-1.awsapprunner.com` (App Runner +
> `db.t4g.micro` RDS + `t4g.small` EC2 Neo4j). Total provisioning + smoke
> ~12 min wall-clock, ~$30/mo.

The canonical Attestor deploy on AWS:

```
       ┌──────────────────────┐         ┌────────────────────────┐
       │ App Runner           │         │ RDS Postgres 16        │
client →│ attestor-api (HTTPS) │ ──TCP──→│ db.t4g.micro + pgvec   │
       └──────────────────────┘   ↑     └────────────────────────┘
                  │                ↑
                  │                │     ┌────────────────────────┐
                  │                └────→│ EC2 t4g.small          │
                  │                      │ neo4j:5.24 (bolt 7687) │
                  ▼                      └────────────────────────┘
       ┌──────────────────────┐
       │ Secrets Manager      │
       │ (5 secrets)          │
       └──────────────────────┘
```

Estimated monthly cost (cheapest tiers): ~**$30/month**
(RDS `db.t4g.micro` ~$15, EC2 `t4g.small` ~$15, App Runner ~$5 idle, free tier covers some).

## 0. Prereqs

```bash
aws configure                       # creds + default region us-east-1
aws sts get-caller-identity         # confirm
```

Native registry for AWS pulls: `public.ecr.aws/m6h5j7o3/attestor:api-4.0.0a5`
(public — no ECR auth needed for App Runner to pull from this).

> **No default VPC?** AWS deletes the default VPC on accounts that never
> launched EC2. Reuse any existing VPC with public-IGW-routed subnets, or
> create one (`aws ec2 create-default-vpc`). On the chosen subnets, ensure
> `MapPublicIpOnLaunch=true` so RDS + EC2 instances get reachable IPs.

## 1. Secrets Manager

```bash
PG_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
NEO4J_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)

for n in postgres-password neo4j-password voyage-api-key openai-api-key openrouter-api-key; do
  case $n in
    postgres-password)   v="$PG_PASSWORD" ;;
    neo4j-password)      v="$NEO4J_PASSWORD" ;;
    voyage-api-key)      v="$VOYAGE_API_KEY" ;;
    openai-api-key)      v="$OPENAI_API_KEY" ;;
    openrouter-api-key)  v="$OPENROUTER_API_KEY" ;;
  esac
  aws secretsmanager create-secret --name $n --secret-string "$v" 2>/dev/null \
    || aws secretsmanager update-secret --secret-id $n --secret-string "$v"
done
```

## 2. RDS Postgres + pgvector

```bash
# Default-VPC security group that allows the App Runner egress + your IP
SG=$(aws ec2 create-security-group --group-name attestor-db-sg \
  --description "attestor postgres + neo4j" \
  --query 'GroupId' --output text 2>/dev/null \
  || aws ec2 describe-security-groups --group-names attestor-db-sg --query 'SecurityGroups[0].GroupId' --output text)

# Open 5432 + 7687 for App Runner (any IPv4 is the simplest demo path; tighten for prod)
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 5432 --cidr 0.0.0.0/0 2>/dev/null
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 7687 --cidr 0.0.0.0/0 2>/dev/null

aws rds create-db-instance \
  --db-instance-identifier attestor-pg \
  --db-instance-class db.t4g.micro \
  --engine postgres --engine-version 16 \
  --master-username postgres --master-user-password "$PG_PASSWORD" \
  --allocated-storage 20 --backup-retention-period 0 \
  --publicly-accessible \
  --vpc-security-group-ids $SG \
  --db-name attestor

aws rds wait db-instance-available --db-instance-identifier attestor-pg

PG_HOST=$(aws rds describe-db-instances --db-instance-identifier attestor-pg \
  --query 'DBInstances[0].Endpoint.Address' --output text)

# pgvector extension
PGPASSWORD="$PG_PASSWORD" psql -h $PG_HOST -U postgres -d attestor -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 3. EC2 Neo4j

```bash
# Latest Amazon Linux 2023 ARM64 AMI
AMI=$(aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=al2023-ami-*-arm64' 'Name=state,Values=available' \
  --query 'sort_by(Images, &CreationDate) | [-1].ImageId' --output text)

# Inline cloud-init: install Docker + run Neo4j
USER_DATA=$(cat <<EOF
#!/bin/bash
dnf install -y docker
systemctl enable --now docker
docker run -d --name neo4j --restart=always \\
  -p 7687:7687 -p 7474:7474 \\
  -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \\
  -e NEO4J_PLUGINS='["graph-data-science"]' \\
  neo4j:5.24-community
EOF
)

aws ec2 run-instances \
  --image-id $AMI --instance-type t4g.small \
  --security-group-ids $SG \
  --user-data "$USER_DATA" \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=attestor-neo4j}]'

# Note: t4g.small (not t4g.micro). Neo4j JVM + GDS plugin needs ~512MB
# heap; t4g.micro at 0.5GB total RAM OOMs during plugin init. Cost delta
# is ~$8/mo; the smaller instance is a false economy.

# Get the public IP once running
INST=$(aws ec2 describe-instances \
  --filters 'Name=tag:Name,Values=attestor-neo4j' 'Name=instance-state-name,Values=running' \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
NEO4J_HOST=$(aws ec2 describe-instances --instance-ids $INST \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
```

## 4. App Runner service

App Runner pulls from ECR Public natively (no ECR connector role needed).

```bash
# Build the apprunner-source.json with all env + secret bindings
cat > /tmp/apprunner-source.json <<EOF
{
  "ImageRepository": {
    "ImageIdentifier": "public.ecr.aws/m6h5j7o3/attestor:api-4.0.0a5",
    "ImageRepositoryType": "ECR_PUBLIC",
    "ImageConfiguration": {
      "Port": "8080",
      "RuntimeEnvironmentVariables": {
        "POSTGRES_URL":             "postgresql://${PG_HOST}:5432",
        "POSTGRES_DATABASE":        "attestor",
        "POSTGRES_USERNAME":        "postgres",
        "POSTGRES_SSLMODE":         "require",
        "NEO4J_URI":                "bolt://${NEO4J_HOST}:7687",
        "NEO4J_USERNAME":           "neo4j",
        "ATTESTOR_V4":              "1",
        "ATTESTOR_DISABLE_LOCAL_EMBED": "1"
      },
      "RuntimeEnvironmentSecrets": {
        "POSTGRES_PASSWORD":     "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:postgres-password",
        "NEO4J_PASSWORD":        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:neo4j-password",
        "VOYAGE_API_KEY":        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:voyage-api-key",
        "OPENAI_API_KEY":        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:openai-api-key",
        "OPENROUTER_API_KEY":    "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:openrouter-api-key"
      }
    }
  },
  "AutoDeploymentsEnabled": false
}
EOF

aws apprunner create-service \
  --service-name attestor-api \
  --source-configuration file:///tmp/apprunner-source.json \
  --instance-configuration Cpu=0.25vCPU,Memory=0.5GB

aws apprunner wait service-running --service-arn \
  $(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='attestor-api'].ServiceArn | [0]" --output text)
```

> Resolve `ACCOUNT` to your real AWS account ID before running. The
> service principal needs `secretsmanager:GetSecretValue` on each secret;
> App Runner creates the role automatically the first time you reference
> a Secrets Manager ARN.

> **Empty secret values are rejected** by Secrets Manager. If your
> `OPENAI_API_KEY` is empty (e.g., you only use OpenRouter), store a
> placeholder string. The smoke does not exercise OpenAI — Voyage handles
> embeddings and the retrieval pipeline is deterministic with no LLM in
> the critical path — so a placeholder is harmless for `smoke_api.py`.

## 5. Smoke

```bash
URL=https://$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='attestor-api'].ServiceUrl | [0]" --output text)
.venv/bin/python scripts/smoke_api.py --url "$URL"
```

## 6. Teardown

```bash
SVC_ARN=$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='attestor-api'].ServiceArn | [0]" --output text)
aws apprunner delete-service --service-arn "$SVC_ARN"

aws rds delete-db-instance --db-instance-identifier attestor-pg --skip-final-snapshot --delete-automated-backups
aws ec2 terminate-instances --instance-ids $INST
aws ec2 delete-security-group --group-id $SG    # after instances stop

for n in postgres-password neo4j-password voyage-api-key openai-api-key openrouter-api-key; do
  aws secretsmanager delete-secret --secret-id $n --force-delete-without-recovery
done
```
