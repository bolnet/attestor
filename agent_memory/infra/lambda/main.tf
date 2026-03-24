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

variable "lambda_memory_mb" {
  default = 512
}

variable "lambda_timeout" {
  default = 60
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

# ── ECR Repository ──

resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "keep_3" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# ── IAM Role ──

resource "aws_iam_role" "lambda" {
  name = "${var.project_name}-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ── Lambda Function ──

resource "aws_lambda_function" "app" {
  function_name = var.project_name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:latest"
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout
  architectures = ["x86_64"]

  environment {
    variables = {
      MEMWRIGHT_DATA_DIR = "/tmp/memwright"
      ARANGO_URL         = var.arango_url
      ARANGO_PASSWORD    = var.arango_password
      ARANGO_DATABASE    = var.arango_database
      ARANGO_TLS_VERIFY  = var.arango_tls_verify
    }
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy_attachment.lambda_basic]

  lifecycle {
    ignore_changes = [image_uri]
  }
}

# ── API Gateway HTTP API ──

resource "aws_apigatewayv2_api" "http" {
  name          = var.project_name
  protocol_type = "HTTP"
  tags          = local.tags
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.app.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
  tags        = local.tags
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# ── Outputs ──

output "api_url" {
  value = aws_apigatewayv2_api.http.api_endpoint
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.app.function_name
}

output "test_command" {
  value = "curl ${aws_apigatewayv2_api.http.api_endpoint}/health"
}
