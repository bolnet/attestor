# Terraform configuration for memwright on GCP AlloyDB
#
# Provisions:
#   - AlloyDB cluster and primary instance
#   - VPC with Private Service Access (required by AlloyDB)
#   - IAM service account with AlloyDB + Vertex AI permissions
#   - Required API enablement
#
# Usage:
#   cd agent_memory/infra
#   terraform init
#   terraform plan -var="project_id=my-project"
#   terraform apply -var="project_id=my-project"

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ── Variables ──

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for AlloyDB cluster"
  type        = string
  default     = "us-central1"
}

variable "cluster_id" {
  description = "AlloyDB cluster ID"
  type        = string
  default     = "memwright-cluster"
}

variable "instance_id" {
  description = "AlloyDB primary instance ID"
  type        = string
  default     = "memwright-primary"
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "memwright"
}

variable "network_name" {
  description = "VPC network name"
  type        = string
  default     = "memwright-vpc"
}

variable "machine_type" {
  description = "Machine type for AlloyDB instance"
  type        = string
  default     = "db-f1-micro"
}

variable "db_password" {
  description = "AlloyDB initial postgres user password"
  type        = string
  sensitive   = true
}

# ── Provider ──

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Enable Required APIs ──

resource "google_project_service" "alloydb" {
  service            = "alloydb.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "aiplatform" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "servicenetworking" {
  service            = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

# ── VPC + Private Service Access ──

resource "google_compute_network" "vpc" {
  name                    = var.network_name
  auto_create_subnetworks = true

  depends_on = [google_project_service.compute]
}

resource "google_compute_global_address" "private_ip" {
  name          = "memwright-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]

  depends_on = [google_project_service.servicenetworking]
}

# ── AlloyDB Cluster ──

resource "google_alloydb_cluster" "main" {
  cluster_id = var.cluster_id
  location   = var.region

  network_config {
    network = google_compute_network.vpc.id
  }

  initial_user {
    user     = "postgres"
    password = var.db_password
  }

  depends_on = [
    google_project_service.alloydb,
    google_service_networking_connection.private_vpc,
  ]
}

# ── AlloyDB Primary Instance ──

resource "google_alloydb_instance" "primary" {
  cluster       = google_alloydb_cluster.main.name
  instance_id   = var.instance_id
  instance_type = "PRIMARY"

  machine_config {
    cpu_count = 2
  }

  database_flags = {
    # Enable pgvector and AGE extensions
    "alloydb.enable_pgvector" = "on"
  }

  depends_on = [google_alloydb_cluster.main]
}

# ── IAM Service Account ──

resource "google_service_account" "memwright" {
  account_id   = "memwright-sa"
  display_name = "Memwright Service Account"
}

resource "google_project_iam_member" "alloydb_client" {
  project = var.project_id
  role    = "roles/alloydb.client"
  member  = "serviceAccount:${google_service_account.memwright.email}"
}

resource "google_project_iam_member" "alloydb_db_user" {
  project = var.project_id
  role    = "roles/alloydb.databaseUser"
  member  = "serviceAccount:${google_service_account.memwright.email}"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.memwright.email}"
}

# ── Outputs ──

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "AlloyDB region"
  value       = var.region
}

output "cluster" {
  description = "AlloyDB cluster ID"
  value       = google_alloydb_cluster.main.cluster_id
}

output "instance" {
  description = "AlloyDB primary instance ID"
  value       = google_alloydb_instance.primary.instance_id
}

output "database" {
  description = "Database name"
  value       = var.database_name
}

output "service_account_email" {
  description = "Service account email for IAM auth"
  value       = google_service_account.memwright.email
}

output "memwright_config" {
  description = "Config dict for GCPBackend"
  value = {
    project_id = var.project_id
    region     = var.region
    cluster    = google_alloydb_cluster.main.cluster_id
    instance   = google_alloydb_instance.primary.instance_id
    database   = var.database_name
  }
}
