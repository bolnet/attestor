terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

variable "azure_location" {
  default = "eastus"
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

provider "azurerm" {
  features {}
}

# ── Resource Group ──

resource "azurerm_resource_group" "app" {
  name     = "${var.project_name}-rg"
  location = var.azure_location
  tags     = local.tags
}

# ── Container Registry ──

resource "azurerm_container_registry" "app" {
  name                = replace(var.project_name, "-", "")
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

# ── Log Analytics (required by Container Apps) ──

resource "azurerm_log_analytics_workspace" "app" {
  name                = "${var.project_name}-logs"
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

# ── Container Apps Environment ──

resource "azurerm_container_app_environment" "app" {
  name                       = "${var.project_name}-env"
  resource_group_name        = azurerm_resource_group.app.name
  location                   = azurerm_resource_group.app.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.app.id
  tags                       = local.tags
}

# ── Container App ──

resource "azurerm_container_app" "app" {
  name                         = var.project_name
  container_app_environment_id = azurerm_container_app_environment.app.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  tags                         = local.tags

  registry {
    server               = azurerm_container_registry.app.login_server
    username             = azurerm_container_registry.app.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.app.admin_password
  }

  secret {
    name  = "arango-password"
    value = var.arango_password
  }

  ingress {
    external_enabled = true
    target_port      = 8080

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 3

    container {
      name   = var.project_name
      image  = "${azurerm_container_registry.app.login_server}/${var.project_name}:latest"
      cpu    = 2.0
      memory = "4Gi"

      env {
        name  = "MEMWRIGHT_DATA_DIR"
        value = "/tmp/memwright"
      }
      env {
        name  = "ARANGO_URL"
        value = var.arango_url
      }
      env {
        name        = "ARANGO_PASSWORD"
        secret_name = "arango-password"
      }
      env {
        name  = "ARANGO_DATABASE"
        value = var.arango_database
      }
      env {
        name  = "ARANGO_TLS_VERIFY"
        value = var.arango_tls_verify
      }

      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8080

        initial_delay    = 5
        interval_seconds = 30
      }

      startup_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8080

        initial_delay    = 5
        interval_seconds = 10
      }
    }
  }
}

# ── Outputs ──

output "service_url" {
  value = "https://${azurerm_container_app.app.latest_revision_fqdn}"
}

output "acr_login_server" {
  value = azurerm_container_registry.app.login_server
}
