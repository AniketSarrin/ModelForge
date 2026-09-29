output "dataset_bucket" { value = aws_s3_bucket.datasets.bucket }
output "cognito_user_pool_id" { value = aws_cognito_user_pool.users.id }
output "cognito_client_id" { value = aws_cognito_user_pool_client.web.id }
output "dataset_metadata_table" { value = aws_dynamodb_table.dataset_metadata.name }
output "usage_table" { value = aws_dynamodb_table.user_usage.name }
