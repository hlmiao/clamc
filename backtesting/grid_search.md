```python
## Step1，更新pandas模块，用于数据导入及整理的模块

!pip install --upgrade pandas -i https://opentuna.cn/pypi/web/simple
!pip install xgboost -i https://opentuna.cn/pypi/web/simple
```

    Looking in indexes: https://opentuna.cn/pypi/web/simple
    Requirement already satisfied: pandas in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (1.1.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas) (2021.1)
    Requirement already satisfied: numpy>=1.15.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas) (1.19.5)
    Requirement already satisfied: six>=1.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    Looking in indexes: https://opentuna.cn/pypi/web/simple
    Collecting xgboost
      Downloading https://opentuna.cn/pypi/web/packages/bb/35/169eec194bf1f9ef52ed670f5032ef2abaf6ed285cfadcb4b6026b800fc9/xgboost-1.4.2-py3-none-manylinux2010_x86_64.whl (166.7 MB)
    [K     |████████████████████████████████| 166.7 MB 10 kB/s s eta 0:00:01
    [?25hRequirement already satisfied: scipy in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from xgboost) (1.5.3)
    Requirement already satisfied: numpy in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from xgboost) (1.19.5)
    Installing collected packages: xgboost
    Successfully installed xgboost-1.4.2



```python
## Step2，导入sys模块，sys模块包含了与Python解释器和环境有关的函数

import sys

directory = '/home/ec2-user/SageMaker/xgboost-new'
if directory not in sys.path:
    sys.path.append(directory)

job_name = 'grid_search'
image_tag = 'latest'
```


```python
## Step3，导入boto3和Sagemaker模块
##在Sagemaker中获取xgboost模型映像的Amazon ECR容器URI

import boto3
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
session = sagemaker.Session()
aws_default_region = session.boto_session.region_name
aws_account_id = session.boto_session.client('sts').get_caller_identity()['Account']
bucket = session.default_bucket()

s3 = boto3.client('s3')
ecs = boto3.client('ecs')

from sagemaker import image_uris
print(image_uris.retrieve(framework='xgboost',region='cn-northwest-1',version='1.2-1'))
```

    451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-xgboost:1.2-1



```python
## Step4，使用get-login-password针对Amazon ECR注册表验证Docker

!aws ecr get-login-password --region {aws_default_region} | docker login --username AWS --password-stdin 451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn
```

    WARNING! Your password will be stored unencrypted in /home/ec2-user/.docker/config.json.
    Configure a credential helper to remove this warning. See
    https://docs.docker.com/engine/reference/commandline/login/#credentials-store
    
    Login Succeeded



```python
## Step5，编辑从官方镜像仓库pull的xgboots镜像，安装boto3，pandas，sklearn和awscli
##对每一个任务都有train_X和train_Y的数据集作为input，然后镜像每次启动后会把train_X和train_Y下载到容器里，然后执行交叉验证

%%writefile {directory}/{job_name}/Dockerfile
FROM 451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-xgboost:1.2-1

RUN pip --no-cache-dir install -i https://opentuna.cn/pypi/web/simple \
        boto3 \
        pandas \
        sklearn

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install && rm awscliv2.zip && rm -rf aws/

ENV ENVIRONMENT="/home/environment"
RUN mkdir -p $ENVIRONMENT/input
RUN mkdir -p $ENVIRONMENT/output
COPY job.py $ENVIRONMENT/
CMD aws s3 cp s3://sagemaker-cn-northwest-1-685095924131/grid_search/$TIMESTAMP/trial_$TRIAL/input/ $ENVIRONMENT/input/ --recursive && python $ENVIRONMENT/job.py && aws s3 cp $ENVIRONMENT/output/ s3://sagemaker-cn-northwest-1-685095924131/grid_search/$TIMESTAMP/trial_$TRIAL/output/ --recursive
```

    Overwriting /home/ec2-user/SageMaker/xgboost-new/grid_search/Dockerfile



```python
## Step6，导入json，pandas模块
##从xgboost.sklearn导入XGBClassifier；从sklearn.model_selection导入cross_val_score
##读取train_X.csv和train_Y.csv文件，利用cross_val_score做交叉验证评分，将评分结果放到output.json
##把xgboots的回归模型做了cross validation（k-ford）

%%writefile {directory}/{job_name}/job.py
import json
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

from timeit import default_timer as timer

__start__ = timer()


with open('/home/environment/input/param.json', 'r') as fin:
    param = json.load(fin)
print(param)
model = XGBClassifier(random_state=0, use_label_encoder=False)
m = model.set_params(**param)
train_X = pd.read_csv('/home/environment/input/train_X.csv', index_col=0)
train_Y = pd.read_csv('/home/environment/input/train_Y.csv', index_col=0)
score = cross_val_score(m, train_X, train_Y, scoring='roc_auc', cv=5, n_jobs=-1).mean()

# 输出
score = {'score': score}
with open('/home/environment/output/output.json', 'w') as fout:
    json.dump(score, fout)


run_time = timer() - __start__
print("Run time %f seconds " % run_time)
```

    Writing /home/ec2-user/SageMaker/xgboost-new/grid_search/job.py



```python
## Step7，创建ECR镜像存储库

!aws ecr create-repository --repository-name {job_name} --region {aws_default_region} > /dev/null
```

    
    An error occurred (RepositoryAlreadyExistsException) when calling the CreateRepository operation: The repository with name 'grid_search' already exists in the registry with id '685095924131'



```python
## Step8，创建新的Docker image

!cd {directory} && docker build {job_name} -t {job_name}:latest
```

    Sending build context to Docker daemon  4.096kB
    Step 1/8 : FROM 451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-xgboost:1.2-1
     ---> 87cbbea7628d
    Step 2/8 : RUN pip --no-cache-dir install -i https://opentuna.cn/pypi/web/simple         boto3         pandas         sklearn
     ---> Using cache
     ---> db7f928804a3
    Step 3/8 : RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install && rm awscliv2.zip && rm -rf aws/
     ---> Using cache
     ---> 5e2de51d229f
    Step 4/8 : ENV ENVIRONMENT="/home/environment"
     ---> Using cache
     ---> ba5c09104151
    Step 5/8 : RUN mkdir -p $ENVIRONMENT/input
     ---> Using cache
     ---> aa9a88ebace2
    Step 6/8 : RUN mkdir -p $ENVIRONMENT/output
     ---> Using cache
     ---> 7efd4a7c402c
    Step 7/8 : COPY job.py $ENVIRONMENT/
     ---> 4ebece4bad0f
    Step 8/8 : CMD aws s3 cp s3://sagemaker-cn-northwest-1-685095924131/grid_search/$TIMESTAMP/trial_$TRIAL/input/ $ENVIRONMENT/input/ --recursive && python $ENVIRONMENT/job.py && aws s3 cp $ENVIRONMENT/output/ s3://sagemaker-cn-northwest-1-685095924131/grid_search/$TIMESTAMP/trial_$TRIAL/output/ --recursive
     ---> Running in 5db61768c772
    Removing intermediate container 5db61768c772
     ---> f67ec1030d0a
    Successfully built f67ec1030d0a
    Successfully tagged grid_search:latest



```python
## Step9，创建新的Docker image并把image push到自己账户的ECR

image_uri = '{}.dkr.ecr.{}.amazonaws.com.cn/{}:{}'.format(aws_account_id, aws_default_region, job_name, 'latest')
exist_image = !docker images -q {job_name}:latest 2> /dev/null
if len(exist_image) > 0:
    !docker tag {job_name}:latest {image_uri}
!$(aws ecr get-login --region {aws_default_region} --no-include-email)
print('Pushing image')
!docker push {image_uri}
print('Done')
```

    WARNING! Using --password via the CLI is insecure. Use --password-stdin.
    WARNING! Your password will be stored unencrypted in /home/ec2-user/.docker/config.json.
    Configure a credential helper to remove this warning. See
    https://docs.docker.com/engine/reference/commandline/login/#credentials-store
    
    Login Succeeded
    Pushing image
    The push refers to repository [685095924131.dkr.ecr.cn-northwest-1.amazonaws.com.cn/grid_search]
    
    [1B6c72cf5e: Preparing 
    [1Bd143f2af: Preparing 
    [1B51cf7f72: Preparing 
    [1Be054e8bc: Preparing 
    [1B025e5b9c: Preparing 
    [1B3cb30e82: Preparing 
    [1B66ed3ac6: Preparing 
    [1B62cc6fa9: Preparing 
    [1Bf7474f5e: Preparing 
    [1Be1a2a6ad: Preparing 
    [1B4969712d: Preparing 
    [1B2126cab2: Preparing 
    [1B4b8590bd: Preparing 
    [1Ba7dfcf02: Preparing 
    [1B02007e9c: Preparing 
    [1B26f01227: Preparing 
    [1B0b24dbe8: Preparing 
    [1B16ac41ff: Preparing 
    [1B39d1c767: Preparing 
    [1B46d7b29d: Preparing 
    [1B5b4f5c34: Preparing 
    [1Ba7535923: Preparing 
    [1Bc2fc7eb9: Preparing 
    [1B3ad0f1b5: Preparing 
    [1B8881187d: Preparing 
    [1B5df75b44: Preparing 
    [9B39d1c767: Pushing  1.286GB/1.642GB[24A[2K[26A[2K[23A[2K[24A[2K[22A[2K[19A[2K[24A[2K[22A[2K[24A[2K[18A[2K[17A[2K[15A[2K[24A[2K[18A[2K[17A[2K[16A[2K[13A[2K[12A[2K[24A[2K[12A[2K[14A[2K[24A[2K[13A[2K[24A[2K[9A[2K[24A[2K[10A[2K[9A[2K[12A[2K[9A[2K[24A[2K[9A[2K[12A[2K[24A[2K[12A[2K[9A[2K[10A[2K[9A[2K[9A[2K[24A[2K[8A[2K[10A[2K[9A[2K[12A[2K[9A[2K[10A[2K[12A[2K[10A[2K[9A[2K[8A[2K[24A[2K[8A[2K[24A[2K[9A[2K[24A[2K[9A[2K[24A[2K[9A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[24A[2K[9A[2K[8A[2K[9A[2K[24A[2K[10A[2K[24A[2K[9A[2K[24A[2K[12A[2K[9A[2K[10A[2K[24A[2K[8A[2K[24A[2K[8A[2K[24A[2K[12A[2K[24A[2K[8A[2K[9A[2K[24A[2K[8A[2K[24A[2K[8A[2K[24A[2K[12A[2K[9A[2K[8A[2K[9A[2K[8A[2K[9A[2K[8A[2K[9A[2K[9A[2K[24A[2K[9A[2K[24A[2K[9A[2K[24A[2K[12A[2K[24A[2K[8A[2K[10A[2K[24A[2K[10A[2K[24A[2K[10A[2K[9A[2K[24A[2K[8A[2K[24A[2K[24A[2K[12A[2K[9A[2K[12A[2K[10A[2K[8A[2K[12A[2K[10A[2K[12A[2K[9A[2K[12A[2K[9A[2K[10A[2K[12A[2K[24A[2K[12A[2K[10A[2K[9A[2K[12A[2K[24A[2K[9A[2K[24A[2K[9A[2K[10A[2K[24A[2K[10A[2K[9A[2K[9A[2K[24A[2K[9A[2K[10A[2K[9A[2K[8A[2K[9A[2K[9A[2K[10A[2K[9A[2K[10A[2K[8A[2K[10A[2K[12A[2K[12A[2K[12A[2K[9A[2K[10A[2K[7A[2K[9A[2K[8A[2K[12A[2K[8A[2K[10A[2K[8A[2K[10A[2K[9A[2K[12A[2K[9A[2K[7A[2K[9A[2K[12A[2K[9A[2K[9A[2K[8A[2K[9A[2K[8A[2K[12A[2K[8A[2K[7A[2K[8A[2K[7A[2K[9A[2K[12A[2K[10A[2K[8A[2K[10A[2K[8A[2K[12A[2K[10A[2K[12A[2K[10A[2K[12A[2K[10A[2K[8A[2K[10A[2K[7A[2K[10A[2K[7A[2K[10A[2K[12A[2K[10A[2K[12A[2K[7A[2K[12A[2K[9A[2K[10A[2K[7A[2K[9A[2K[10A[2K[10A[2K[8A[2K[9A[2K[12A[2K[9A[2K[10A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[9A[2K[12A[2K[10A[2K[7A[2K[8A[2K[10A[2K[8A[2K[10A[2K[7A[2K[10A[2K[7A[2K[9A[2K[8A[2K[9A[2K[8A[2K[10A[2K[8A[2K[10A[2K[7A[2K[10A[2K[7A[2K[8A[2K[10A[2K[12A[2K[10A[2K[12A[2K[10A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[9A[2K[8A[2K[12A[2K[9A[2K[12A[2K[7A[2K[10A[2K[7A[2K[10A[2K[7A[2K[8A[2K[10A[2K[12A[2K[7A[2K[12A[2K[10A[2K[12A[2K[7A[2K[12A[2K[8A[2K[12A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[10A[2K[9A[2K[12A[2K[9A[2K[7A[2K[8A[2K[7A[2K[9A[2K[7A[2K[12A[2K[8A[2K[10A[2K[10A[2K[9A[2K[12A[2K[8A[2K[10A[2K[12A[2K[8A[2K[9A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[7A[2K[12A[2K[9A[2K[12A[2K[9A[2K[10A[2K[8A[2K[7A[2K[8A[2K[9A[2K[8A[2K[9A[2K[12A[2K[8A[2K[10A[2K[8A[2K[10A[2K[8A[2K[9A[2K[12A[2K[9A[2K[10A[2K[9A[2K[7A[2K[12A[2K[7A[2K[8A[2K[7A[2K[12A[2K[7A[2K[10A[2K[7A[2K[8A[2K[7A[2K[8A[2K[9A[2K[8A[2K[9A[2K[8A[2K[7A[2K[9A[2K[10A[2K[7A[2K[8A[2K[9A[2K[7A[2K[9A[2K[10A[2K[9A[2K[12A[2K[7A[2K[8A[2K[10A[2K[9A[2K[9A[2K[7A[2K[10A[2K[9A[2K[10A[2K[12A[2K[10A[2K[9A[2K[8A[2K[9A[2K[8A[2K[9A[2K[8A[2K[9A[2K[8A[2K[10A[2K[7A[2K[10A[2K[9A[2K[8A[2K[9A[2K[12A[2K[8A[2K[10A[2K[8A[2K[7A[2K[8A[2K[7A[2K[10A[2K[7A[2K[10A[2K[9A[2K[10A[2K[12A[2K[7A[2K[12A[2K[9A[2K[12A[2K[9A[2K[12A[2K[7A[2K[12A[2K[10A[2K[12A[2K[10A[2K[12A[2K[10A[2K[9A[2K[10A[2K[8A[2K[10A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[12A[2K[7A[2K[12A[2K[9A[2K[7A[2K[9A[2K[7A[2K[10A[2K[12A[2K[6A[2K[12A[2K[10A[2K[12A[2K[12A[2K[9A[2K[12A[2K[7A[2K[12A[2K[10A[2K[12A[2K[7A[2K[6A[2K[10A[2K[12A[2K[9A[2K[12A[2K[9A[2K[12A[2K[9A[2K[7A[2K[10A[2K[5A[2K[12A[2K[9A[2K[12A[2K[10A[2K[5A[2K[10A[2K[7A[2K[10A[2K[9A[2K[5A[2K[9A[2K[12A[2K[9A[2K[12A[2K[7A[2K[9A[2K[7A[2K[7A[2K[10A[2K[10A[2K[10A[2K[7A[2K[10A[2K[7A[2K[5A[2K[12A[2K[5A[2K[9A[2K[5A[2K[9A[2K[5A[2K[10A[2K[5A[2K[10A[2K[12A[2K[5A[2K[9A[2K[5A[2K[9A[2K[10A[2K[9A[2K[12A[2K[9A[2K[12A[2K[7A[2K[10A[2K[7A[2K[10A[2K[9A[2K[9A[2K[12A[2K[9A[2K[12A[2K[9A[2K[7A[2K[9A[2K[10A[2K[12A[2K[10A[2K[12A[2K[9A[2K[12A[2K[9A[2K[7A[2K[10A[2K[10A[2K[12A[2K[10A[2K[7A[2K[10A[2K[10A[2K[10A[2K[12A[2K[4A[2K[10A[2K[7A[2K[10A[2K[7A[2K[9A[2K[7A[2K[9A[2K[10A[2K[9A[2K[10A[2K[12A[2K[4A[2K[7A[2K[4A[2K[7A[2K[9A[2K[10A[2K[9A[2K[10A[2K[9A[2K[12A[2K[9A[2K[10A[2K[7A[2K[10A[2K[12A[2K[9A[2K[12A[2K[7A[2K[12A[2K[7A[2K[12A[2K[10A[2K[9A[2K[12A[2K[9A[2K[12A[2K[10A[2K[12A[2K[7A[2K[9A[2K[7A[2K[12A[2K[7A[2K[12A[2K[7A[2K[7A[2K[10A[2K[7A[2K[9A[2K[3A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[10A[2K[9A[2K[12A[2K[7A[2K[12A[2K[7A[2K[9A[2K[10A[2K[9A[2K[10A[2K[9A[2K[10A[2K[10A[2K[12A[2K[10A[2K[7A[2K[10A[2K[12A[2K[10A[2K[10A[2K[9A[2K[2A[2K[9A[2K[12A[2K[1A[2K[12A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[1A[2K[7A[2K[12A[2K[7A[2K[1A[2K[7A[2K[12A[2K[12A[2K[12A[2K[1A[2K[12A[2K[12A[2K[7A[2K[1A[2K[7A[2K[9A[2K[1A[2K[7A[2K[1A[2K[7A[2K[9A[2K[7A[2K[9A[2K[12A[2K[9A[2K[7A[2K[7A[2K[1A[2K[7A[2K[1A[2K[7A[2K[1A[2K[7A[2K[9A[2K[7A[2K[1A[2K[7A[2K[9A[2K[7A[2K[1A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[1A[2K[9A[2K[1A[2K[1A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[1A[2K[9A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[7A[2K[9A[2K[9A[2K[9A[2K[7A[2K[7A[2K[9A[2K[9A[2KPushing  646.1MB/800.7MB[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9B39d1c767: Pushed   1.694GB/1.642GB[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2Klatest: digest: sha256:43044de104b0378a45a882ebafac38c08e54e7786d84c50dcdf94175e8c7789a size: 5965
    Done



```python
## Step10，导入datetime，json，OS，time，numpy，pandas，XGBClassifier和cross_val_score

import datetime
import json
import os
import time
import numpy as np
import pandas as pd
from functools import reduce
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

from timeit import default_timer as timer
_i_ = 0
```


```python
!pwd
```

    /home/ec2-user/SageMaker



```python
## Step11，获取当前时间，读取文件“风格因子数据”和“timelist_m“，打印出运行的时长，判断哪部分运行时间最长

__start__ = timer()


path = directory + '/'
df_all = pd.read_csv(path + '风格因子数据.csv', converters={'stcode': str}, index_col=0)
timelist_m = pd.read_csv(path + 'timelist_m.csv', header=None, squeeze=True)
timelist_m = pd.to_datetime(timelist_m)
df_all.index = pd.to_datetime(df_all.index)


run_time = timer() - __start__
_i_ += 1
print(u'Part', _i_, '|', "run time %f seconds " % run_time)
```

    Part 1 | run time 9.796672 seconds 



```python
__start__ = timer()


#%% 分类算法
df_all['y_pred'] = np.nan
df_all['y_pred_rank'] = np.nan
df_all['group'] = np.nan
factor_list = ['规模', '估值', '分红', '盈利', '财务质量', '成长', '反转', '波动率', '流动性', '分析师预期变化']
importance = pd.DataFrame(index=timelist_m[: -1], columns=factor_list)
auc = pd.DataFrame(index=timelist_m[: -1], columns=['auc'])
window = 24


run_time = timer() - __start__
_i_ += 1
print(u'Part', _i_, '|', "run time %f seconds " % run_time)
```

    Part 2 | run time 0.091344 seconds 



```python
__start__ = timer()


for i in range(len(timelist_m[: -1]) - window):
    #i=0
    timestamp = datetime.datetime.strftime(timelist_m[i], '%Y-%m-%d')
    df_used = df_all.loc[timelist_m[i : i + window]]
    df_temp = pd.DataFrame(df_used[df_used['flag'] != 0])
    df_temp['flag'] = ((df_temp['flag']+ 1)/2).astype(int).values
    train_X = df_temp[factor_list]
    mean = train_X.mean()
    std = train_X.std()
    train_X = (train_X - mean) / std # 标准化
    train_Y = df_temp['flag']
    
    ## 定义参数值
    model = XGBClassifier(random_state=0, use_label_encoder=False)
    params = {'n_estimators': [50, 250, 500], 'max_depth': [4, 7, 10], 'min_child_weight': [0.2, 0.6, 1], 'gamma': [0.2, 1]}
    
    ## 两两组合构成参数集（54种参数）
    param_keys = list(params)
    param_values = list(params.values())
    param_values[0] = [[x] for x in param_values[0]]
    combined_values = reduce(lambda x, y: [i + [j] for i in x for j in y], param_values)
    param_list = [dict(zip(param_keys, x)) for x in combined_values]
    task_list = []
    
    ## Step12，遍历参数集寻找最优参数，把train_x，train_y放到s3，启动容器运行计算任务，再把score数值以json格式放到s3；
    ## 然后再选择score数值最高的参数组合，放回到pandas的一个结果表
    print('开始参数搜索', timestamp, '，共', len(param_list), '组参数:', end = '')
    for i, param in enumerate(param_list):
        s3.put_object(Body=json.dumps(param), Bucket=bucket, Key='{}/{}/trial_{}/input/param.json'.format(job_name, timestamp, str(i)))
        s3.put_object(Body=train_X.to_csv(), Bucket=bucket, Key='{}/{}/trial_{}/input/train_X.csv'.format(job_name, timestamp, str(i)))
        s3.put_object(Body=train_Y.to_csv(), Bucket=bucket, Key='{}/{}/trial_{}/input/train_Y.csv'.format(job_name, timestamp, str(i)))
        r = ecs.run_task(
            cluster='test',
            enableExecuteCommand=False,
            group='family:grid_search',
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        'subnet-1564535f',
                        'subnet-b440f6dd',
                        'subnet-15a66d6e',
                    ],
                    'securityGroups': [
                        'sg-7683ba1e',
                    ],
                    'assignPublicIp': 'ENABLED'
                }
            },
            overrides={
                'containerOverrides': [
                    {
                        'name': 'clwm',
                        'environment': [
                            {
                                'name': 'TIMESTAMP',
                                'value': timestamp
                            },
                            {
                                'name': 'TRIAL',
                                'value': str(i)
                            }
                        ]
                    },
                ]
            },
            platformVersion='1.4.0',
            taskDefinition='arn:aws-cn:ecs:cn-northwest-1:685095924131:task-definition/grid_search:2'
        )
        task_list.append(r['tasks'][0]['taskArn'])
    
    # track 任务的执行状态
    r = ecs.describe_tasks(cluster='test', tasks=task_list)
    stopped_count = 0
    while stopped_count < len(param_list):
        pending_count = 0
        running_count = 0
        stopped_count = 0
        for i in range(len(r['tasks'])):
            if r['tasks'][i]['lastStatus'] == 'PENDING':
                pending_count += 1
            elif r['tasks'][i]['lastStatus'] == 'RUNNING':
                running_count += 1
            elif r['tasks'][i]['lastStatus'] == 'STOPPED':
                stopped_count += 1
        print('PENDING JOBS:', pending_count, 'RUNNING JOBS:', running_count, 'STOPPED JOBS:', stopped_count)
        time.sleep(10)
        r = ecs.describe_tasks(cluster='test', tasks=task_list)
        
    # 获取 score
    score = []
    for i, param in enumerate(param_list):
        key = '{}/{}/trial_{}/output/output.json'.format(job_name, timestamp, str(i))
        score.append(json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())['score'])
    
    # 训练模型
    best_score = np.max(score)
    best_param = param_list[np.argmax(score)]
    print('Best param:', best_param, 'Score:', best_score)
    model.set_params(**best_param)
    model_cv = model.fit(train_X, train_Y)
    
    # 输出因子数据
    temp_X_test = df_all[factor_list].loc[timelist_m[i + window]]
    temp_X_test = (temp_X_test - mean) / std # 标准化
    
    auc.loc[timelist_m[i + window - 1], :] = best_score
    importance.loc[timelist_m[i + window - 1], :] = model_cv.feature_importances_
    y_pred_temp = pd.Series(model_cv.predict_proba(temp_X_test)[: , 1], index=temp_X_test.index)
    df_all.loc[timelist_m[i + window], 'y_pred'] = y_pred_temp
    df_all.loc[timelist_m[i + window], 'y_pred_rank'] = y_pred_temp.rank()
    df_all.loc[timelist_m[i + window], 'group'] = pd.qcut(df_all.loc[timelist_m[i + window], 'y_pred'], np.arange(0, 1.1, 0.1), labels=np.arange(1, 11)).astype(int)


run_time = timer() - __start__
_i_ += 1
print(u'Part', _i_, '|', "run time %f seconds " % run_time)
```

    开始参数搜索 2010-01-29 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 44 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 38 RUNNING JOBS: 10 STOPPED JOBS: 2
    PENDING JOBS: 31 RUNNING JOBS: 13 STOPPED JOBS: 5
    PENDING JOBS: 27 RUNNING JOBS: 15 STOPPED JOBS: 9
    PENDING JOBS: 19 RUNNING JOBS: 19 STOPPED JOBS: 12
    PENDING JOBS: 15 RUNNING JOBS: 23 STOPPED JOBS: 13
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 9 RUNNING JOBS: 21 STOPPED JOBS: 18
    PENDING JOBS: 8 RUNNING JOBS: 21 STOPPED JOBS: 24
    PENDING JOBS: 3 RUNNING JOBS: 17 STOPPED JOBS: 24
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 33
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5713225394001743
    [11:12:22] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-02-26 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 1 STOPPED JOBS: 0
    PENDING JOBS: 47 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 44 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 9 STOPPED JOBS: 1
    PENDING JOBS: 33 RUNNING JOBS: 13 STOPPED JOBS: 5
    PENDING JOBS: 28 RUNNING JOBS: 15 STOPPED JOBS: 8
    PENDING JOBS: 21 RUNNING JOBS: 20 STOPPED JOBS: 11
    PENDING JOBS: 14 RUNNING JOBS: 24 STOPPED JOBS: 12
    PENDING JOBS: 11 RUNNING JOBS: 26 STOPPED JOBS: 16
    PENDING JOBS: 10 RUNNING JOBS: 21 STOPPED JOBS: 17
    PENDING JOBS: 7 RUNNING JOBS: 22 STOPPED JOBS: 22
    PENDING JOBS: 4 RUNNING JOBS: 17 STOPPED JOBS: 25
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 29
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 1} Score: 0.5782052493314891
    [11:18:47] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-03-31 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 43 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 38 RUNNING JOBS: 10 STOPPED JOBS: 2
    PENDING JOBS: 32 RUNNING JOBS: 12 STOPPED JOBS: 5
    PENDING JOBS: 28 RUNNING JOBS: 14 STOPPED JOBS: 9
    PENDING JOBS: 20 RUNNING JOBS: 21 STOPPED JOBS: 11
    PENDING JOBS: 14 RUNNING JOBS: 24 STOPPED JOBS: 12
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 9 RUNNING JOBS: 21 STOPPED JOBS: 16
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 24
    PENDING JOBS: 3 RUNNING JOBS: 18 STOPPED JOBS: 24
    PENDING JOBS: 3 RUNNING JOBS: 16 STOPPED JOBS: 30
    PENDING JOBS: 1 RUNNING JOBS: 17 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 0.2} Score: 0.5743566110920579
    [11:25:13] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-04-30 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 36 RUNNING JOBS: 12 STOPPED JOBS: 2
    PENDING JOBS: 31 RUNNING JOBS: 13 STOPPED JOBS: 5
    PENDING JOBS: 27 RUNNING JOBS: 15 STOPPED JOBS: 9
    PENDING JOBS: 20 RUNNING JOBS: 18 STOPPED JOBS: 12
    PENDING JOBS: 13 RUNNING JOBS: 24 STOPPED JOBS: 15
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 8 RUNNING JOBS: 22 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 23
    PENDING JOBS: 3 RUNNING JOBS: 17 STOPPED JOBS: 26
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 33
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 1} Score: 0.5711188252127275
    [11:31:43] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-05-31 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 46 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 37 RUNNING JOBS: 11 STOPPED JOBS: 2
    PENDING JOBS: 33 RUNNING JOBS: 11 STOPPED JOBS: 6
    PENDING JOBS: 25 RUNNING JOBS: 17 STOPPED JOBS: 8
    PENDING JOBS: 21 RUNNING JOBS: 17 STOPPED JOBS: 12
    PENDING JOBS: 13 RUNNING JOBS: 24 STOPPED JOBS: 13
    PENDING JOBS: 11 RUNNING JOBS: 24 STOPPED JOBS: 16
    PENDING JOBS: 9 RUNNING JOBS: 22 STOPPED JOBS: 19
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 22
    PENDING JOBS: 4 RUNNING JOBS: 18 STOPPED JOBS: 25
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 14 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 39
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 0.2} Score: 0.5802897412256457
    [11:38:22] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-06-30 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 36 RUNNING JOBS: 12 STOPPED JOBS: 2
    PENDING JOBS: 31 RUNNING JOBS: 13 STOPPED JOBS: 6
    PENDING JOBS: 27 RUNNING JOBS: 15 STOPPED JOBS: 9
    PENDING JOBS: 21 RUNNING JOBS: 19 STOPPED JOBS: 12
    PENDING JOBS: 15 RUNNING JOBS: 23 STOPPED JOBS: 13
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 7 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 5 RUNNING JOBS: 24 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 20 STOPPED JOBS: 24
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 30
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 39
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5841504994529889
    [11:45:01] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-07-30 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 46 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 1} Score: 0.581751118818809
    [11:58:13] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-09-30 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 44 RUNNING JOBS: 10 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 37 RUNNING JOBS: 10 STOPPED JOBS: 3
    PENDING JOBS: 32 RUNNING JOBS: 11 STOPPED JOBS: 6
    PENDING JOBS: 26 RUNNING JOBS: 16 STOPPED JOBS: 10
    PENDING JOBS: 20 RUNNING JOBS: 20 STOPPED JOBS: 12
    PENDING JOBS: 15 RUNNING JOBS: 23 STOPPED JOBS: 13
    PENDING JOBS: 11 RUNNING JOBS: 24 STOPPED JOBS: 16
    PENDING JOBS: 8 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 23 STOPPED JOBS: 21
    PENDING JOBS: 4 RUNNING JOBS: 19 STOPPED JOBS: 24
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 27
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 1} Score: 0.582432525779865
    [12:05:05] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2010-10-29 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 46 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 38 RUNNING JOBS: 9 STOPPED JOBS: 3
    PENDING JOBS: 32 RUNNING JOBS: 11 STOPPED JOBS: 6
    PENDING JOBS: 25 RUNNING JOBS: 17 STOPPED JOBS: 10
    PENDING JOBS: 19 RUNNING JOBS: 21 STOPPED JOBS: 12
    PENDING JOBS: 14 RUNNING JOBS: 23 STOPPED JOBS: 13
    PENDING JOBS: 10 RUNNING JOBS: 26 STOPPED JOBS: 17
    PENDING JOBS: 8 RUNNING JOBS: 23 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 23 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 21 STOPPED JOBS: 25
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 29
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 15 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 15 RUNNING JOBS: 22 STOPPED JOBS: 13
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 10 RUNNING JOBS: 23 STOPPED JOBS: 18
    PENDING JOBS: 7 RUNNING JOBS: 22 STOPPED JOBS: 21
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 24
    PENDING JOBS: 3 RUNNING JOBS: 16 STOPPED JOBS: 26
    PENDING JOBS: 1 RUNNING JOBS: 17 STOPPED JOBS: 34
    PENDING JOBS: 1 RUNNING JOBS: 17 STOPPED JOBS: 35
    PENDING JOBS: 1 RUNNING JOBS: 15 STOPPED JOBS: 36
    PENDING JOBS: 1 RUNNING JOBS: 12 STOPPED JOBS: 36
    PENDING JOBS: 1 RUNNING JOBS: 12 STOPPED JOBS: 40
    PENDING JOBS: 1 RUNNING JOBS: 8 STOPPED JOBS: 41
    PENDING JOBS: 1 RUNNING JOBS: 7 STOPPED JOBS: 44
    PENDING JOBS: 1 RUNNING JOBS: 5 STOPPED JOBS: 45
    PENDING JOBS: 1 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 1 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 49
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 1 RUNNING JOBS: 1 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 0.2} Score: 0.5817845222517739
    [12:26:20] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-01-31 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 46 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 36 RUNNING JOBS: 10 STOPPED JOBS: 4
    PENDING JOBS: 30 RUNNING JOBS: 14 STOPPED JOBS: 6
    PENDING JOBS: 27 RUNNING JOBS: 15 STOPPED JOBS: 9
    PENDING JOBS: 17 RUNNING JOBS: 24 STOPPED JOBS: 12
    PENDING JOBS: 13 RUNNING JOBS: 24 STOPPED JOBS: 13
    PENDING JOBS: 10 RUNNING JOBS: 25 STOPPED JOBS: 17
    PENDING JOBS: 8 RUNNING JOBS: 24 STOPPED JOBS: 19
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 23 STOPPED JOBS: 25
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 27
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 1} Score: 0.5784597105674061
    [12:33:04] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-02-28 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 45 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 37 RUNNING JOBS: 9 STOPPED JOBS: 4
    PENDING JOBS: 32 RUNNING JOBS: 12 STOPPED JOBS: 6
    PENDING JOBS: 26 RUNNING JOBS: 16 STOPPED JOBS: 10
    PENDING JOBS: 19 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 26 STOPPED JOBS: 13
    PENDING JOBS: 9 RUNNING JOBS: 27 STOPPED JOBS: 16
    PENDING JOBS: 8 RUNNING JOBS: 23 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 20
    PENDING JOBS: 4 RUNNING JOBS: 23 STOPPED JOBS: 26
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 27
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5889870450121796
    [12:39:58] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-03-31 ，共 54 组参数:PENDING JOBS: 45 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 53 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 45 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 40 RUNNING JOBS: 11 STOPPED JOBS: 0
    PENDING JOBS: 35 RUNNING JOBS: 11 STOPPED JOBS: 3
    PENDING JOBS: 32 RUNNING JOBS: 12 STOPPED JOBS: 7
    PENDING JOBS: 26 RUNNING JOBS: 16 STOPPED JOBS: 9
    PENDING JOBS: 21 RUNNING JOBS: 19 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 14
    PENDING JOBS: 10 RUNNING JOBS: 26 STOPPED JOBS: 17
    PENDING JOBS: 9 RUNNING JOBS: 22 STOPPED JOBS: 17
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 23 STOPPED JOBS: 25
    PENDING JOBS: 1 RUNNING JOBS: 17 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 0.2} Score: 0.5891377471787317
    [12:46:52] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-04-29 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 45 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 40 RUNNING JOBS: 10 STOPPED JOBS: 0
    PENDING JOBS: 35 RUNNING JOBS: 12 STOPPED JOBS: 4
    PENDING JOBS: 30 RUNNING JOBS: 14 STOPPED JOBS: 7
    PENDING JOBS: 27 RUNNING JOBS: 15 STOPPED JOBS: 10
    PENDING JOBS: 18 RUNNING JOBS: 21 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 14
    PENDING JOBS: 9 RUNNING JOBS: 27 STOPPED JOBS: 15
    PENDING JOBS: 8 RUNNING JOBS: 22 STOPPED JOBS: 18
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 24
    PENDING JOBS: 3 RUNNING JOBS: 24 STOPPED JOBS: 26
    PENDING JOBS: 2 RUNNING JOBS: 16 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 34
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 1} Score: 0.5890673105420207
    [12:53:58] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-05-31 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 44 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 40 RUNNING JOBS: 10 STOPPED JOBS: 0
    PENDING JOBS: 34 RUNNING JOBS: 12 STOPPED JOBS: 4
    PENDING JOBS: 30 RUNNING JOBS: 13 STOPPED JOBS: 8
    PENDING JOBS: 25 RUNNING JOBS: 17 STOPPED JOBS: 10
    PENDING JOBS: 16 RUNNING JOBS: 24 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 14
    PENDING JOBS: 11 RUNNING JOBS: 23 STOPPED JOBS: 17
    PENDING JOBS: 9 RUNNING JOBS: 21 STOPPED JOBS: 19
    PENDING JOBS: 7 RUNNING JOBS: 21 STOPPED JOBS: 22
    PENDING JOBS: 5 RUNNING JOBS: 22 STOPPED JOBS: 26
    PENDING JOBS: 1 RUNNING JOBS: 19 STOPPED JOBS: 27
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 33
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 14 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 40
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 0.2} Score: 0.5816195548497897
    [13:01:06] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-06-30 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5839380733188198
    [13:22:30] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-09-30 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 1 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 45 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 11 STOPPED JOBS: 1
    PENDING JOBS: 34 RUNNING JOBS: 12 STOPPED JOBS: 4
    PENDING JOBS: 32 RUNNING JOBS: 10 STOPPED JOBS: 8
    PENDING JOBS: 24 RUNNING JOBS: 18 STOPPED JOBS: 9
    PENDING JOBS: 17 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 14
    PENDING JOBS: 9 RUNNING JOBS: 25 STOPPED JOBS: 17
    PENDING JOBS: 7 RUNNING JOBS: 24 STOPPED JOBS: 19
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 22
    PENDING JOBS: 2 RUNNING JOBS: 23 STOPPED JOBS: 25
    PENDING JOBS: 1 RUNNING JOBS: 20 STOPPED JOBS: 27
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 33
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 15 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5777016310441772
    [13:29:49] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-10-31 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 10 STOPPED JOBS: 0
    PENDING JOBS: 37 RUNNING JOBS: 11 STOPPED JOBS: 2
    PENDING JOBS: 35 RUNNING JOBS: 11 STOPPED JOBS: 5
    PENDING JOBS: 30 RUNNING JOBS: 12 STOPPED JOBS: 8
    PENDING JOBS: 23 RUNNING JOBS: 19 STOPPED JOBS: 12
    PENDING JOBS: 17 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 13 RUNNING JOBS: 23 STOPPED JOBS: 15
    PENDING JOBS: 9 RUNNING JOBS: 25 STOPPED JOBS: 17
    PENDING JOBS: 8 RUNNING JOBS: 23 STOPPED JOBS: 19
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 23 STOPPED JOBS: 26
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 15 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 0.2} Score: 0.5820715233145523
    [13:37:10] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2011-11-30 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 52 RUNNING JOBS: 1 STOPPED JOBS: 0
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 25
    PENDING JOBS: 1 RUNNING JOBS: 19 STOPPED JOBS: 25
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 17 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 39
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 49 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 47 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 10 STOPPED JOBS: 4
    PENDING JOBS: 33 RUNNING JOBS: 13 STOPPED JOBS: 5
    PENDING JOBS: 31 RUNNING JOBS: 11 STOPPED JOBS: 7
    PENDING JOBS: 25 RUNNING JOBS: 17 STOPPED JOBS: 12
    PENDING JOBS: 17 RUNNING JOBS: 23 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 24 STOPPED JOBS: 13
    PENDING JOBS: 10 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 9 RUNNING JOBS: 22 STOPPED JOBS: 19
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 22
    PENDING JOBS: 3 RUNNING JOBS: 24 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 21 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 40
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 1} Score: 0.583613758726425
    [14:35:15] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-07-31 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 49 RUNNING JOBS: 5 STOPPED JOBS: 0
    PENDING JOBS: 43 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 10 STOPPED JOBS: 1
    PENDING JOBS: 34 RUNNING JOBS: 13 STOPPED JOBS: 4
    PENDING JOBS: 31 RUNNING JOBS: 12 STOPPED JOBS: 7
    PENDING JOBS: 24 RUNNING JOBS: 18 STOPPED JOBS: 11
    PENDING JOBS: 19 RUNNING JOBS: 21 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 14
    PENDING JOBS: 11 RUNNING JOBS: 25 STOPPED JOBS: 17
    PENDING JOBS: 8 RUNNING JOBS: 25 STOPPED JOBS: 17
    PENDING JOBS: 6 RUNNING JOBS: 23 STOPPED JOBS: 19
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 25
    PENDING JOBS: 3 RUNNING JOBS: 19 STOPPED JOBS: 25
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 31
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 35
    PENDING JOBS: 2 RUNNING JOBS: 17 STOPPED JOBS: 35
    PENDING JOBS: 2 RUNNING JOBS: 15 STOPPED JOBS: 35
    PENDING JOBS: 2 RUNNING JOBS: 11 STOPPED JOBS: 37
    PENDING JOBS: 2 RUNNING JOBS: 10 STOPPED JOBS: 41
    PENDING JOBS: 2 RUNNING JOBS: 9 STOPPED JOBS: 41
    PENDING JOBS: 2 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 2 RUNNING JOBS: 7 STOPPED JOBS: 44
    PENDING JOBS: 2 RUNNING JOBS: 4 STOPPED JOBS: 44
    PENDING JOBS: 2 RUNNING JOBS: 3 STOPPED JOBS: 47
    PENDING JOBS: 2 RUNNING JOBS: 2 STOPPED JOBS: 48
    PENDING JOBS: 1 RUNNING JOBS: 3 STOPPED JOBS: 49
    PENDING JOBS: 1 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 1 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5868638959839065
    [14:43:49] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-08-31 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 49 RUNNING JOBS: 4 STOPPED JOBS: 0
    PENDING JOBS: 46 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 8 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 9 STOPPED JOBS: 2
    PENDING JOBS: 35 RUNNING JOBS: 10 STOPPED JOBS: 5
    PENDING JOBS: 29 RUNNING JOBS: 13 STOPPED JOBS: 8
    PENDING JOBS: 24 RUNNING JOBS: 18 STOPPED JOBS: 12
    PENDING JOBS: 17 RUNNING JOBS: 23 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 24 STOPPED JOBS: 14
    PENDING JOBS: 11 RUNNING JOBS: 23 STOPPED JOBS: 18
    PENDING JOBS: 9 RUNNING JOBS: 21 STOPPED JOBS: 19
    PENDING JOBS: 6 RUNNING JOBS: 23 STOPPED JOBS: 22
    PENDING JOBS: 4 RUNNING JOBS: 24 STOPPED JOBS: 25
    PENDING JOBS: 2 RUNNING JOBS: 20 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 19 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 37
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5849543298385408
    [14:51:03] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-09-28 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 50 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 47 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 8 STOPPED JOBS: 2
    PENDING JOBS: 34 RUNNING JOBS: 12 STOPPED JOBS: 6
    PENDING JOBS: 30 RUNNING JOBS: 12 STOPPED JOBS: 8
    PENDING JOBS: 24 RUNNING JOBS: 18 STOPPED JOBS: 11
    PENDING JOBS: 17 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 11 RUNNING JOBS: 26 STOPPED JOBS: 15
    PENDING JOBS: 9 RUNNING JOBS: 26 STOPPED JOBS: 17
    PENDING JOBS: 7 RUNNING JOBS: 22 STOPPED JOBS: 18
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 23
    PENDING JOBS: 3 RUNNING JOBS: 25 STOPPED JOBS: 25
    PENDING JOBS: 1 RUNNING JOBS: 19 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 33
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 35
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 15 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 39
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 48
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.2, 'gamma': 0.2} Score: 0.590428363908655
    [14:58:36] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-10-31 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 49 RUNNING JOBS: 3 STOPPED JOBS: 0
    PENDING JOBS: 47 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 42 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 38 RUNNING JOBS: 9 STOPPED JOBS: 3
    PENDING JOBS: 33 RUNNING JOBS: 12 STOPPED JOBS: 6
    PENDING JOBS: 30 RUNNING JOBS: 12 STOPPED JOBS: 8
    PENDING JOBS: 25 RUNNING JOBS: 16 STOPPED JOBS: 11
    PENDING JOBS: 16 RUNNING JOBS: 23 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 15
    PENDING JOBS: 10 RUNNING JOBS: 26 STOPPED JOBS: 17
    PENDING JOBS: 7 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 21
    PENDING JOBS: 4 RUNNING JOBS: 24 STOPPED JOBS: 26
    PENDING JOBS: 1 RUNNING JOBS: 18 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 32
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 41
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 51
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 0.2} Score: 0.5898724573473935
    [15:06:00] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-11-30 ，共 54 组参数:PENDING JOBS: 47 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 2 STOPPED JOBS: 0
    PENDING JOBS: 47 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 41 RUNNING JOBS: 11 STOPPED JOBS: 0
    PENDING JOBS: 39 RUNNING JOBS: 9 STOPPED JOBS: 2
    PENDING JOBS: 34 RUNNING JOBS: 12 STOPPED JOBS: 6
    PENDING JOBS: 30 RUNNING JOBS: 12 STOPPED JOBS: 8
    PENDING JOBS: 22 RUNNING JOBS: 20 STOPPED JOBS: 12
    PENDING JOBS: 17 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 24 STOPPED JOBS: 15
    PENDING JOBS: 10 RUNNING JOBS: 25 STOPPED JOBS: 18
    PENDING JOBS: 8 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 5 RUNNING JOBS: 23 STOPPED JOBS: 21
    PENDING JOBS: 3 RUNNING JOBS: 25 STOPPED JOBS: 26
    PENDING JOBS: 1 RUNNING JOBS: 19 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 10 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 44
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 0.6, 'gamma': 1} Score: 0.5876624991700886
    [15:13:22] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2012-12-31 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 1 STOPPED JOBS: 0
    PENDING JOBS: 49 RUNNING JOBS: 5 STOPPED JOBS: 0
    PENDING JOBS: 44 RUNNING JOBS: 7 STOPPED JOBS: 0
    PENDING JOBS: 40 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 35 RUNNING JOBS: 11 STOPPED JOBS: 4
    PENDING JOBS: 31 RUNNING JOBS: 12 STOPPED JOBS: 8
    PENDING JOBS: 26 RUNNING JOBS: 16 STOPPED JOBS: 10
    PENDING JOBS: 18 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 12 RUNNING JOBS: 25 STOPPED JOBS: 13
    PENDING JOBS: 10 RUNNING JOBS: 25 STOPPED JOBS: 16
    PENDING JOBS: 8 RUNNING JOBS: 24 STOPPED JOBS: 18
    PENDING JOBS: 6 RUNNING JOBS: 22 STOPPED JOBS: 20
    PENDING JOBS: 4 RUNNING JOBS: 24 STOPPED JOBS: 25
    PENDING JOBS: 2 RUNNING JOBS: 20 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 13 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 40
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 6 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 47
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 3 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 0.2} Score: 0.5818640897195834
    [15:20:51] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2013-01-31 ，共 54 组参数:PENDING JOBS: 46 RUNNING JOBS: 0 STOPPED JOBS: 0
    PENDING JOBS: 51 RUNNING JOBS: 1 STOPPED JOBS: 0
    PENDING JOBS: 48 RUNNING JOBS: 6 STOPPED JOBS: 0
    PENDING JOBS: 43 RUNNING JOBS: 9 STOPPED JOBS: 0
    PENDING JOBS: 40 RUNNING JOBS: 9 STOPPED JOBS: 1
    PENDING JOBS: 34 RUNNING JOBS: 13 STOPPED JOBS: 5
    PENDING JOBS: 31 RUNNING JOBS: 11 STOPPED JOBS: 6
    PENDING JOBS: 24 RUNNING JOBS: 18 STOPPED JOBS: 11
    PENDING JOBS: 18 RUNNING JOBS: 22 STOPPED JOBS: 12
    PENDING JOBS: 14 RUNNING JOBS: 22 STOPPED JOBS: 14
    PENDING JOBS: 10 RUNNING JOBS: 26 STOPPED JOBS: 16
    PENDING JOBS: 8 RUNNING JOBS: 23 STOPPED JOBS: 18
    PENDING JOBS: 5 RUNNING JOBS: 24 STOPPED JOBS: 21
    PENDING JOBS: 3 RUNNING JOBS: 25 STOPPED JOBS: 25
    PENDING JOBS: 0 RUNNING JOBS: 22 STOPPED JOBS: 26
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 31
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 18 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 16 STOPPED JOBS: 36
    PENDING JOBS: 0 RUNNING JOBS: 12 STOPPED JOBS: 38
    PENDING JOBS: 0 RUNNING JOBS: 11 STOPPED JOBS: 42
    PENDING JOBS: 0 RUNNING JOBS: 9 STOPPED JOBS: 43
    PENDING JOBS: 0 RUNNING JOBS: 8 STOPPED JOBS: 45
    PENDING JOBS: 0 RUNNING JOBS: 7 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 5 STOPPED JOBS: 46
    PENDING JOBS: 0 RUNNING JOBS: 4 STOPPED JOBS: 49
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 50
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 2 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 1 STOPPED JOBS: 52
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 53
    PENDING JOBS: 0 RUNNING JOBS: 0 STOPPED JOBS: 54
    Best param: {'n_estimators': 50, 'max_depth': 4, 'min_child_weight': 1, 'gamma': 1} Score: 0.5859479231163708
    [15:28:10] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    开始参数搜索 2013-02-28 ，共 54 组参数:


```python
## Step13，最后再把结果保存到sagemaker的路径下，进行查看
__start__ = timer()


df_m = df_all.loc[timelist_m[: -1]].dropna(subset=['group'])
df_m.reset_index()[['trade_dt', 'stcode', 'y_pred']].to_csv(directory + '/机器学习合成因子_Xgboost.csv')


run_time = timer() - __start__
_i_ += 1
print(u'Part', _i_, '|', "run time %f seconds " % run_time)
```

    Part 4 | run time 1.465199 seconds 

