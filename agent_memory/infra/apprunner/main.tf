terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "memwright"
}

variable "arango_url" {
  description = "ArangoDB Oasis endpoint URL"
  sensitive   = true
}

variable "arango_password" {
  description = "ArangoDB root password"
  sensitive   = true
}

variable "arango_database" {
  default = "memwright"
}

variable "arango_tls_verify" {
  default = "false"
}

locals {
  tags = {
    project    = var.project_name
    managed_by = "terraform"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Use existing ECR repo from Lambda deploy ──

data "aws_ecr_repository" "app" {
  name = var.project_name
}

# ── IAM: App Runner ECR access role ──

resource "aws_iam_role" "apprunner_ecr" {
  name = "${var.project_name}-apprunner-ecr"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# ── App Runner Service ──

resource "aws_apprunner_service" "app" {
  service_name = var.project_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr.arn
    }

    image_repository {
      image_identifier      = "${data.aws_ecr_repository.app.repository_url}:apprunner"
      image_repository_type = "ECR"

      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          MEMWRIGHT_DATA_DIR = "/tmp/memwright"
          ARANGO_URL         = var.arango_url
          ARANGO_PASSWORD    = var.arango_password
          ARANGO_DATABASE    = var.arango_database
          ARANGO_TLS_VERIFY  = var.arango_tls_verify
        }
      }
    }

    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu    = "2048"
    memory = "4096"
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr]
}

# ── Outputs ──

output "service_url" {
  value = "https://${aws_apprunner_service.app.service_url}"
}

output "service_arn" {
  value = aws_apprunner_service.app.arn
}

output "service_status" {
  value = aws_apprunner_service.app.status
}
