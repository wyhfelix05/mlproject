# 使用 AWS 官方 Lambda Python 3.10 基础镜像
FROM public.ecr.aws/lambda/python:3.10

# 安装依赖到 /var/task
COPY requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt -t /var/task

# 设置工作目录
WORKDIR /var/task

# 复制应用代码
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY lambda_handler.py ./

# 设置 Lambda 入口函数：文件名.函数名
CMD ["lambda_handler.lambda_handler"]
