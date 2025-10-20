# Start from official 3.11 image which will have Python 3.11 installed and other extra packages
FROM python:3.11

# Install system-level dependencies needed by experiment
RUN apt-get update && apt-get install -y \
    git wget curl build-essential make cmake pkg-config python3-dev \
    libxml2-dev libxslt1-dev poppler-utils qpdf mupdf-tools

# Copy from root repo into the container
COPY PythonDependencies.txt /PythonDependencies.txt
COPY entrypoint.sh /entrypoint.sh

# Upgrade and install pip and install all Python packages listed in the files.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /PythonDependencies.txt


# /entrypoint.sh is main program/script from root repo that will always run when container
# starts.
ENTRYPOINT ["/entrypoint.sh"]
