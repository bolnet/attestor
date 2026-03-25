terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "gcp_project" {
  description = "GCP project ID"
}

variable "gcp_region" {
  default = "us-central1"
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
  labels = {
    project    = var.project_name
    managed-by = "terraform"
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ── Artifact Registry ──

resource "google_artifact_registry_repository" "app" {
  location      = var.gcp_region
  repository_id = var.project_name
  format        = "DOCKER"
  labels        = local.labels
}

# ── Cloud Run Service ──

resource "google_cloud_run_v2_service" "app" {
  name     = var.project_name
  location = var.gcp_region
  labels   = local.labels

  template {
    containers {
      image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.project_name}/${var.project_name}:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "MEMWRIGHT_DATA_DIR"
        value = "/tmp/memwright"
      }
      env {
        name  = "ARANGO_URL"
        value = var.arango_url
      }
      env {
        name  = "ARANGO_PASSWORD"
        value = var.arango_password
      }
      env {
        name  = "ARANGO_DATABASE"
        value = var.arango_database
      }
      env {
        name  = "ARANGO_TLS_VERIFY"
        value = var.arango_tls_verify
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  depends_on = [google_artifact_registry_repository.app]
}

# ── Allow unauthenticated access ──

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.gcp_project
  location = var.gcp_region
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Outputs ──

output "service_url" {
  value = google_cloud_run_v2_service.app.uri
}

output "artifact_registry_url" {
  value = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project}/${var.project_name}"
}
