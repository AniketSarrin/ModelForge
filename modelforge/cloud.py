from __future__ import annotations
import os


def cloud_config() -> dict:
    """Return non-secret production cloud configuration.

    Credentials are intentionally never read into project files. When AWS_MODE=1,
    normal AWS credential resolution (profile, env vars, IAM role) is used by boto3.
    """
    enabled = os.environ.get('MODELFORGE_AWS_MODE','0') == '1'
    return {
        'enabled': enabled,
        'region': os.environ.get('MODELFORGE_AWS_REGION','us-west-2'),
        'bucket': os.environ.get('MODELFORGE_DATASET_BUCKET',''),
        'cognito_user_pool_id': os.environ.get('MODELFORGE_COGNITO_USER_POOL_ID',''),
        'cognito_client_id': os.environ.get('MODELFORGE_COGNITO_CLIENT_ID',''),
        'metadata_table': os.environ.get('MODELFORGE_DATASET_TABLE',''),
        'usage_table': os.environ.get('MODELFORGE_USAGE_TABLE',''),
        'backend': 'aws' if enabled else 'local',
    }


def s3_client():
    import boto3
    cfg=cloud_config()
    return boto3.client('s3',region_name=cfg['region'])


def presign_upload(user_id:str, dataset_id:str, filename:str, content_type:str='application/octet-stream', expires:int=900)->dict:
    cfg=cloud_config()
    if not cfg['enabled'] or not cfg['bucket']:
        raise RuntimeError('AWS dataset mode is not configured')
    key=f'users/{user_id}/datasets/{dataset_id}/{filename}'
    url=s3_client().generate_presigned_url('put_object',Params={'Bucket':cfg['bucket'],'Key':key,'ContentType':content_type},ExpiresIn=expires)
    return {'url':url,'bucket':cfg['bucket'],'key':key,'expires_in':expires}
