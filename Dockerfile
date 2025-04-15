FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including those needed for Intel oneAPI
RUN apt-get update && \
    apt-get install -y git build-essential wget apt-transport-https gnupg ca-certificates && \
    # Add Intel repository for oneAPI
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    # Install Intel MKL (Math Kernel Library) which accelerates numpy, scipy, etc.
    apt-get install -y intel-oneapi-mkl && \
    # Install poetry
    pip install poetry && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Intel oneAPI
ENV PYTHONPATH=${PYTHONPATH}:/opt/intel/oneapi/lib
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/oneapi/lib:/opt/intel/oneapi/mkl/latest/lib/intel64
ENV LIBRARY_PATH=${LIBRARY_PATH}:/opt/intel/oneapi/lib:/opt/intel/oneapi/mkl/latest/lib/intel64
ENV MKLROOT=/opt/intel/oneapi/mkl/latest

# Copy dependency definition first for better caching
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi

# Copy project files
COPY . .

# Install any other dependencies specifically for daal4py and scikit-learn-intelex
RUN pip install --no-cache-dir daal4py scikit-learn-intelex

# Run application
CMD ["poetry", "run", "python", "src/cli.py"]
