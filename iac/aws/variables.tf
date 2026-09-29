variable "aws_region" { type = string default = "us-west-2" }
variable "project_name" { type = string default = "modelforge" }
variable "environment" { type = string default = "prod" }
variable "cors_origins" { type = list(string) default = ["http://localhost:8000"] }
