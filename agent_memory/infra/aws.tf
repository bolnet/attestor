# Memwright AWS Infrastructure — DynamoDB + OpenSearch Serverless + Neptune Serverless
#
# Usage:
#   terraform init && terraform apply -var="project_name=memwright"
#
# Outputs connection config for AWSBackend.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "project_name" {
  type    = string
  default = "memwright"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "dev"
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  prefix = "${var.project_name}-${var.environment}"
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ═══════════════════════════════════════════════════════════════════════
# DynamoDB — Document Store
# ═══════════════════════════════════════════════════════════════════════

resource "aws_dynamodb_table" "memories" {
  name         = "${var.project_name}_memories"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
  attribute {
    name = "status"
    type = "S"
  }
  attribute {
    name = "category"
    type = "S"
  }
  attribute {
    name = "entity"
    type = "S"
  }
  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "status-created_at-index"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "category-created_at-index"
    hash_key        = "category"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "entity-created_at-index"
    hash_key        = "entity"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = local.tags
}

# ═══════════════════════════════════════════════════════════════════════
# OpenSearch Serverless — Vector Store
# ═══════════════════════════════════════════════════════════════════════

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${local.prefix}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules = [{
      ResourceType = "collection"
      Resource      = ["collection/${local.prefix}-vectors"]
    }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name = "${local.prefix}-net"
  type = "network"
  policy = jsonencode([{
    Rules = [{
      ResourceType = "collection"
      Resource      = ["collection/${local.prefix}-vectors"]
    }, {
      ResourceType = "dashboard"
      Resource      = ["collection/${local.prefix}-vectors"]
    }]
    AllowFromPublic = true
  }])
}

resource "aws_opensearchserverless_access_policy" "data" {
  name = "${local.prefix}-data"
  type = "data"
  policy = jsonencode([{
    Rules = [{
      ResourceType = "index"
      Resource      = ["index/${local.prefix}-vectors/*"]
      Permission   = [
        "aoss:CreateIndex",
        "aoss:UpdateIndex",
        "aoss:DescribeIndex",
        "aoss:ReadDocument",
        "aoss:WriteDocument",
      ]
    }, {
      ResourceType = "collection"
      Resource      = ["collection/${local.prefix}-vectors"]
      Permission   = [
        "aoss:CreateCollectionItems",
        "aoss:DescribeCollectionItems",
        "aoss:UpdateCollectionItems",
      ]
    }]
    Principal = [aws_iam_role.memwright.arn]
  }])
}

resource "aws_opensearchserverless_collection" "vectors" {
  name = "${local.prefix}-vectors"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
    aws_opensearchserverless_access_policy.data,
  ]

  tags = local.tags
}

# ═══════════════════════════════════════════════════════════════════════
# Neptune Serverless — Graph Store
# ═══════════════════════════════════════════════════════════════════════

resource "aws_neptune_cluster" "graph" {
  cluster_identifier                   = "${local.prefix}-neptune"
  engine                               = "neptune"
  engine_version                       = "1.3.2.1"
  iam_database_authentication_enabled  = true
  storage_encrypted                    = true
  skip_final_snapshot                  = true
  apply_immediately                    = true

  serverless_v2_scaling_configuration {
    min_capacity = 1.0
    max_capacity = 8.0
  }

  vpc_security_group_ids = [aws_security_group.neptune.id]
  db_subnet_group_name   = aws_neptune_subnet_group.main.name

  tags = local.tags
}

resource "aws_neptune_cluster_instance" "serverless" {
  identifier           = "${local.prefix}-neptune-inst"
  cluster_identifier   = aws_neptune_cluster.graph.id
  instance_class       = "db.serverless"
  neptune_subnet_group_name = aws_neptune_subnet_group.main.name

  tags = local.tags
}

# Neptune networking — uses default VPC for simplicity
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_neptune_subnet_group" "main" {
  name       = "${local.prefix}-neptune-subnets"
  subnet_ids = data.aws_subnets.default.ids
  tags       = local.tags
}

resource "aws_security_group" "neptune" {
  name_prefix = "${local.prefix}-neptune-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8182
    to_port     = 8182
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# ═══════════════════════════════════════════════════════════════════════
# IAM Role — Least-privilege for memwright
# ═══════════════════════════════════════════════════════════════════════

resource "aws_iam_role" "memwright" {
  name = "${local.prefix}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        AWS = data.aws_caller_identity.current.arn
      }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "dynamodb" {
  name = "dynamodb-access"
  role = aws_iam_role.memwright.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:BatchWriteItem",
        "dynamodb:BatchGetItem",
        "dynamodb:DescribeTable",
        "dynamodb:CreateTable",
      ]
      Resource = [
        aws_dynamodb_table.memories.arn,
        "${aws_dynamodb_table.memories.arn}/index/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "opensearch" {
  name = "opensearch-access"
  role = aws_iam_role.memwright.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["aoss:APIAccessAll"]
      Resource = [aws_opensearchserverless_collection.vectors.arn]
    }]
  })
}

resource "aws_iam_role_policy" "neptune" {
  name = "neptune-access"
  role = aws_iam_role.memwright.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "neptune-db:ReadDataViaQuery",
        "neptune-db:WriteDataViaQuery",
        "neptune-db:DeleteDataViaQuery",
        "neptune-db:GetQueryStatus",
      ]
      Resource = "${aws_neptune_cluster.graph.arn}/*"
    }]
  })
}

resource "aws_iam_role_policy" "bedrock" {
  name = "bedrock-embeddings"
  role = aws_iam_role.memwright.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel"]
      Resource = "arn:aws:bedrock:${var.region}::foundation-model/amazon.titan-embed-text-v2:0"
    }]
  })
}

# ═══════════════════════════════════════════════════════════════════════
# Outputs — feed directly into AWSBackend config
# ═══════════════════════════════════════════════════════════════════════

output "aws_backend_config" {
  description = "Config dict for AWSBackend.__init__()"
  value = {
    region = var.region
    dynamodb = {
      table_prefix = var.project_name
    }
    opensearch = {
      endpoint = aws_opensearchserverless_collection.vectors.collection_endpoint
      index    = "memories"
    }
    neptune = {
      endpoint = aws_neptune_cluster.graph.endpoint
    }
  }
}

output "iam_role_arn" {
  description = "IAM role ARN to assume for memwright access"
  value       = aws_iam_role.memwright.arn
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.memories.name
}

output "opensearch_endpoint" {
  value = aws_opensearchserverless_collection.vectors.collection_endpoint
}

output "neptune_endpoint" {
  value = aws_neptune_cluster.graph.endpoint
}
