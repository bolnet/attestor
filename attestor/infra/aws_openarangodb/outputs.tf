output "alb_url" {
  description = "Memwright API endpoint"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for memwright image"
  value       = aws_ecr_repository.memwright.repository_url
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.memwright.name
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "efs_id" {
  description = "EFS filesystem ID for ArangoDB data"
  value       = aws_efs_file_system.arangodb.id
}
