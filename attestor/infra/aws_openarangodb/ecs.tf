# ═══════════════════════════════════════════════════════════════════════
# ECS Cluster
# ═══════════════════════════════════════════════════════════════════════

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${local.name_prefix}-cluster" }
}

# ═══════════════════════════════════════════════════════════════════════
# IAM — ECS task execution + task role
# ═══════════════════════════════════════════════════════════════════════

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name_prefix}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

# ═══════════════════════════════════════════════════════════════════════
# CloudWatch Log Groups
# ═══════════════════════════════════════════════════════════════════════

resource "aws_cloudwatch_log_group" "attestor" {
  name              = "/ecs/${local.name_prefix}/attestor"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "arangodb" {
  name              = "/ecs/${local.name_prefix}/arangodb"
  retention_in_days = 14
}

# ═══════════════════════════════════════════════════════════════════════
# ECS Task Definition — attestor + ArangoDB sidecar
# ═══════════════════════════════════════════════════════════════════════

resource "aws_ecs_task_definition" "attestor" {
  family                   = "${local.name_prefix}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "arangodb-data"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.arangodb.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    # ── ArangoDB sidecar ──
    {
      name      = "arangodb"
      image     = "arangodb/arangodb:3.12"
      essential = true
      cpu       = var.arangodb_cpu
      memory    = var.arangodb_memory

      entryPoint = ["sh", "-c"]
      command    = ["rm -f /var/lib/arangodb3/LOCK && /entrypoint.sh arangod"]

      portMappings = [{
        containerPort = 8529
        protocol      = "tcp"
      }]

      environment = [
        { name = "ARANGO_NO_AUTH", value = var.arango_password == "" ? "1" : "0" },
        { name = "ARANGO_ROOT_PASSWORD", value = var.arango_password },
      ]

      mountPoints = [{
        sourceVolume  = "arangodb-data"
        containerPath = "/var/lib/arangodb3"
        readOnly      = false
      }]

      healthCheck = {
        command     = ["CMD-SHELL", "wget -qO- http://localhost:8529/_api/version || exit 1"]
        interval    = 10
        timeout     = 5
        retries     = 5
        startPeriod = 30
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.arangodb.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "arangodb"
        }
      }
    },

    # ── Attestor app ──
    {
      name      = "attestor"
      image     = "${aws_ecr_repository.attestor.repository_url}:latest"
      essential = true
      cpu       = var.attestor_cpu
      memory    = var.attestor_memory

      portMappings = [{
        containerPort = 8000
        protocol      = "tcp"
      }]

      environment = [
        { name = "ARANGO_URL", value = "http://localhost:8529" },
        { name = "ARANGO_DATABASE", value = var.arango_database },
        { name = "ARANGO_PASSWORD", value = var.arango_password },
        { name = "ARANGO_TLS_VERIFY", value = "false" },
        { name = "ATTESTOR_DATA_DIR", value = "/data/attestor" },
      ]

      dependsOn = [{
        containerName = "arangodb"
        condition     = "HEALTHY"
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.attestor.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "attestor"
        }
      }
    }
  ])

  tags = { Name = "${local.name_prefix}-task" }
}

# ═══════════════════════════════════════════════════════════════════════
# ECS Service
# ═══════════════════════════════════════════════════════════════════════

resource "aws_ecs_service" "attestor" {
  name            = "${local.name_prefix}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.attestor.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.attestor.arn
    container_name   = "attestor"
    container_port   = 8000
  }

  depends_on = [
    aws_lb_listener.http,
    aws_efs_mount_target.arangodb,
  ]

  tags = { Name = "${local.name_prefix}-service" }
}
