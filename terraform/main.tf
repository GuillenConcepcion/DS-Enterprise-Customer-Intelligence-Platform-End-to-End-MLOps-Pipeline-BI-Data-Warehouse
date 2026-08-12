# MLOps Enterprise Architecture - Terraform Configuration
# This infrastructure defines the AWS components described in the Cloud Topology diagram.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# --- VARIABLES ---
variable "aws_region" {
  type        = string
  description = "Target AWS region for resources"
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Target execution environment"
  default     = "Production"
}

variable "project_name" {
  type        = string
  description = "Name of the project"
  default     = "CustomerChurnML"
}

variable "data_lake_bucket_name" {
  type        = string
  description = "Unique name for the raw data lake S3 bucket"
  default     = "enterprise-mlops-raw-data-bucket-concepcion"
}

variable "ecr_image_identifier" {
  type        = string
  description = "ECR image identifier for App Runner FastAPI serving"
  default     = "public.ecr.aws/myregistry/mlops-churn-api:latest"
}

variable "sagemaker_instance_type" {
  type        = string
  description = "Compute instance type for model development notebook"
  default     = "ml.t3.medium"
}

# --- PROVIDER ---
provider "aws" {
  region = var.aws_region
}

# --- LOCAL TAGS ---
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Owner       = "Guillén Concepción"
  }
}

# --- 1. DATA LAKE (S3) FOR RAW DATA ---
resource "aws_s3_bucket" "mlops_data_lake" {
  bucket        = var.data_lake_bucket_name
  force_destroy = false

  tags = local.common_tags
}

# Enable S3 Bucket Versioning for data governance
resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.mlops_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable S3 Server-side Encryption by default
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake_encryption" {
  bucket = aws_s3_bucket.mlops_data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Explicit Public Access Block (Security best practice)
resource "aws_s3_bucket_public_access_block" "data_lake_public_block" {
  bucket = aws_s3_bucket.mlops_data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- 2. AWS APP RUNNER FOR FASTAPI SERVING ---
resource "aws_apprunner_service" "mlops_fastapi_serving" {
  service_name = "${lower(var.project_name)}-api"

  source_configuration {
    image_repository {
      image_configuration {
        port = "8000"
      }
      image_identifier      = var.ecr_image_identifier
      image_repository_type = "ECR_PUBLIC"
    }
  }

  tags = local.common_tags
}

# --- 3. SAGEMAKER NOTEBOOK INSTANCE ---
resource "aws_sagemaker_notebook_instance" "mlops_sagemaker" {
  name          = "${lower(var.project_name)}-exploration"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  instance_type = var.sagemaker_instance_type

  tags = local.common_tags
}

# IAM Role for SageMaker Notebook execution
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker_execution_role_${lower(var.project_name)}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Attach SageMaker access policies to the execution role
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# --- OUTPUTS ---
output "data_lake_s3_bucket_arn" {
  value       = aws_s3_bucket.mlops_data_lake.arn
  description = "The ARN of the Data Lake S3 bucket"
}

output "app_runner_service_url" {
  value       = aws_apprunner_service.mlops_fastapi_serving.service_url
  description = "The URL of the deployed FastAPI serving endpoint"
}

output "sagemaker_notebook_url" {
  value       = aws_sagemaker_notebook_instance.mlops_sagemaker.url
  description = "The URL of the SageMaker Notebook instance"
}

