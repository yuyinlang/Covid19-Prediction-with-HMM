FROM python:3.7

# Set Working Directory
WORKDIR /usr/src/app

# Install app dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
