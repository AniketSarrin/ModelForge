data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "datasets" {
  bucket_prefix = "${var.project_name}-${var.environment}-datasets-"
  force_destroy = false
  tags = { Project = var.project_name, Environment = var.environment }
}
resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  block_public_acls = true
  block_public_policy = true
  ignore_public_acls = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}
resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  rule {
    id = "abort-incomplete-uploads"
    status = "Enabled"
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}
resource "aws_s3_bucket_cors_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET","PUT","POST","HEAD"]
    allowed_origins = var.cors_origins
    expose_headers = ["ETag"]
    max_age_seconds = 3600
  }
}

resource "aws_cognito_user_pool" "users" {
  name = "${var.project_name}-${var.environment}-users"
  username_attributes = ["email"]
  auto_verified_attributes = ["email"]
  password_policy {
    minimum_length = 8
    require_lowercase = true
    require_numbers = true
    require_uppercase = false
    require_symbols = false
  }
}
resource "aws_cognito_user_pool_client" "web" {
  name = "${var.project_name}-${var.environment}-web"
  user_pool_id = aws_cognito_user_pool.users.id
  generate_secret = false
  explicit_auth_flows = ["ALLOW_USER_SRP_AUTH","ALLOW_USER_PASSWORD_AUTH","ALLOW_REFRESH_TOKEN_AUTH"]
  prevent_user_existence_errors = "ENABLED"
}

resource "aws_dynamodb_table" "dataset_metadata" {
  name = "${var.project_name}-${var.environment}-datasets"
  billing_mode = "PAY_PER_REQUEST"
  hash_key = "user_id"
  range_key = "dataset_id"
  attribute { name = "user_id" type = "S" }
  attribute { name = "dataset_id" type = "S" }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
  tags = { Project = var.project_name, Environment = var.environment }
}

resource "aws_dynamodb_table" "user_usage" {
  name = "${var.project_name}-${var.environment}-usage"
  billing_mode = "PAY_PER_REQUEST"
  hash_key = "user_id"
  attribute { name = "user_id" type = "S" }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}
