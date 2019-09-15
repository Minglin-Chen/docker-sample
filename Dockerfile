# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory
WORKDIR /docker_sample

# Copy the current directory contents into the container
COPY . /docker_sample

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run when the container launches
CMD ["python", "deploy_models.py"]