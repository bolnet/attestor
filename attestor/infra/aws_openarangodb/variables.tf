variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "memwright"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ── ECS ──────────────────────────────────────────────────────────────

variable "app_cpu" {
  description = "CPU units for the entire ECS task (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "app_memory" {
  description = "Memory (MiB) for the entire ECS task"
  type        = number
  default     = 4096
}

variable "memwright_cpu" {
  description = "CPU units for the memwright container"
  type        = number
  default     = 1024
}

variable "memwright_memory" {
  description = "Memory (MiB) for the memwright container"
  type        = number
  default     = 1536
}

variable "arangodb_cpu" {
  description = "CPU units for the ArangoDB sidecar"
  type        = number
  default     = 1024
}

variable "arangodb_memory" {
  description = "Memory (MiB) for the ArangoDB sidecar"
  type        = number
  default     = 2560
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

# ── ArangoDB ─────────────────────────────────────────────────────────

variable "arango_database" {
  description = "ArangoDB database name"
  type        = string
  default     = "memwright"
}

variable "arango_password" {
  description = "ArangoDB root password (empty = no auth)"
  type        = string
  default     = ""
  sensitive   = true
}

# ── Networking ───────────────────────────────────────────────────────

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets (need 2 for ALB)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}
