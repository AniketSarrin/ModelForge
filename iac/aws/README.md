# ModelForge AWS infrastructure

This Terraform stack creates the production primitives for the Advanced data workspace:

- private, encrypted, versioned S3 dataset bucket
- Cognito user pool + public web client
- DynamoDB dataset metadata table
- DynamoDB per-user usage table

The application enforces the 1 GiB/user quota before upload/presigning. S3 does not provide a native per-prefix quota.

No AWS credentials are embedded. Configure your normal AWS CLI/profile or CI role later, then run:

```bash
terraform init
terraform plan
terraform apply
```

Local development uses SQLite + local disk so the complete dataset workflow can be tested without AWS.
