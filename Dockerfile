# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch

# Set the working directory
WORKDIR /docker_sample

# Copy the current directory contents into the container
COPY . /docker_sample

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run when the container launches
CMD ["python", "deploy_models.py"]
