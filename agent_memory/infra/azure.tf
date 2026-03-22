# Azure Cosmos DB infrastructure for Memwright
#
# Provisions:
#   - Cosmos DB account (serverless capacity)
#   - Database and containers with vector index policies
#   - Optional Azure OpenAI resource with text-embedding-3-small
#
# Usage:
#   terraform init
#   terraform plan -var="resource_group_name=memwright-rg" -var="location=eastus"
#   terraform apply -var="resource_group_name=memwright-rg" -var="location=eastus"

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# ═══════════════════════════════════════════════════════════════════════
# Variables
# ═══════════════════════════════════════════════════════════════════════

variable "resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = "memwright-rg"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "cosmos_account_name" {
  description = "Cosmos DB account name (globally unique)"
  type        = string
  default     = "memwright-cosmos"
}

variable "cosmos_database_name" {
  description = "Cosmos DB database name"
  type        = string
  default     = "memwright"
}

variable "enable_openai" {
  description = "Deploy Azure OpenAI resource for embeddings"
  type        = bool
  default     = false
}

variable "openai_account_name" {
  description = "Azure OpenAI account name (globally unique)"
  type        = string
  default     = "memwright-openai"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    project     = "memwright"
    managed_by  = "terraform"
  }
}

# ═══════════════════════════════════════════════════════════════════════
# Resource Group
# ═══════════════════════════════════════════════════════════════════════

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# ═══════════════════════════════════════════════════════════════════════
# Cosmos DB Account (Serverless)
# ═══════════════════════════════════════════════════════════════════════

resource "azurerm_cosmosdb_account" "main" {
  name                = var.cosmos_account_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"
  tags                = var.tags

  capacity {
    total_throughput_limit = -1 # Serverless
  }

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  capabilities {
    name = "EnableServerless"
  }

  # Enable vector search capability
  capabilities {
    name = "EnableNoSQLVectorSearch"
  }
}

# ═══════════════════════════════════════════════════════════════════════
# Cosmos DB Database
# ═══════════════════════════════════════════════════════════════════════

resource "azurerm_cosmosdb_sql_database" "main" {
  name                = var.cosmos_database_name
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
}

# ═══════════════════════════════════════════════════════════════════════
# Cosmos DB Containers
# ═══════════════════════════════════════════════════════════════════════

# Memories container — documents + vector embeddings
resource "azurerm_cosmosdb_sql_container" "memories" {
  name                = "memories"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_paths = ["/category"]

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/embedding/*"
    }
  }

  # Note: Vector index policies must be configured via Azure CLI or SDK
  # after container creation, as Terraform azurerm provider does not yet
  # support vector_embedding_policy natively.
  #
  # Post-apply step:
  #   az cosmosdb sql container update \
  #     --account-name <account> --resource-group <rg> \
  #     --database-name memwright --name memories \
  #     --idx '{"indexingMode":"consistent","automatic":true,
  #       "includedPaths":[{"path":"/*"}],
  #       "excludedPaths":[{"path":"/embedding/*"}],
  #       "vectorIndexes":[{"path":"/embedding","type":"diskANN"}]}'
}

# Graph entities container
resource "azurerm_cosmosdb_sql_container" "graph_entities" {
  name                = "graph_entities"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_paths = ["/entity_type"]

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}

# Graph edges container
resource "azurerm_cosmosdb_sql_container" "graph_edges" {
  name                = "graph_edges"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_paths = ["/from_key"]

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}

# ═══════════════════════════════════════════════════════════════════════
# Azure OpenAI (Optional)
# ═══════════════════════════════════════════════════════════════════════

resource "azurerm_cognitive_account" "openai" {
  count               = var.enable_openai ? 1 : 0
  name                = var.openai_account_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"
  tags                = var.tags
}

resource "azurerm_cognitive_deployment" "embedding" {
  count                = var.enable_openai ? 1 : 0
  name                 = "text-embedding-3-small"
  cognitive_account_id = azurerm_cognitive_account.openai[0].id

  model {
    format  = "OpenAI"
    name    = "text-embedding-3-small"
    version = "1"
  }

  sku {
    name     = "Standard"
    capacity = 120 # 120K tokens per minute
  }
}

# ═══════════════════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════════════════

output "cosmos_endpoint" {
  description = "Cosmos DB account endpoint"
  value       = azurerm_cosmosdb_account.main.endpoint
}

output "cosmos_primary_key" {
  description = "Cosmos DB primary key"
  value       = azurerm_cosmosdb_account.main.primary_key
  sensitive   = true
}

output "cosmos_database" {
  description = "Cosmos DB database name"
  value       = azurerm_cosmosdb_sql_database.main.name
}

output "memwright_config" {
  description = "Memwright backend config (paste into config.json)"
  value = jsonencode({
    cosmos_endpoint = azurerm_cosmosdb_account.main.endpoint
    cosmos_key      = "$AZURE_COSMOS_KEY"
    cosmos_database = azurerm_cosmosdb_sql_database.main.name
  })
}

output "openai_endpoint" {
  description = "Azure OpenAI endpoint (if enabled)"
  value       = var.enable_openai ? azurerm_cognitive_account.openai[0].endpoint : ""
}

output "openai_key" {
  description = "Azure OpenAI key (if enabled)"
  value       = var.enable_openai ? azurerm_cognitive_account.openai[0].primary_access_key : ""
  sensitive   = true
}
