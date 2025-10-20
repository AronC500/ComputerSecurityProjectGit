# PDF Malware Analysis Experiment

### Purpose of this Repository  
- This project attempts to reproduce the work of Liu et al., "Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model" https://arxiv.org/html/2506.17162v1.


### Software Requirements/Packages(Not all needed, but were used Liu's et al. experiment):
- Docker https://www.docker.com/get-started/ (Let you run applications in isolated containers so it basically creates a container that have its own Python installation, libraries, system tools, file system which allow everyone working on the project run the code in the same environment and avoiding missing dependencies or version problems).
- Python 3.11+ (Preferred as it's the one used in Liu's' et al. experiment and some dependencies may require newer versions so it is better to use 3.11+)   
- Rest of python dependencies in ComputerSecurityProjectGit/PythonDependencies.txt.
- System level dependencies: git, wget, curl, build-essential, make, cmake, pkg-config, python3-dev, libxml2-dev, libxslt1-dev, poppler-utils, qpdf, mupdf-tools, unzip.  
- In Docker, use an Docker image that already have Python 3.11 installed. For the python dependencies, install them inside the Dockerfile using 'pip'. For the system-level dependencies, install them in the Dockerfile using 'apt-get'.

### Keyterms you may need to understand this repository:  
- Object Reference Graph(ORG): graph representation of a PDF document.
- Intermediate Representation(IR): an way to represent a PDF's content that makes it easier for program to understand and thus analyze.
- BERT: deep learning model that can be trained using large amount of text.
- Graph Isomorphism Network(GIN): classify graphs or predict properties of nodes/graphs 
- node embeddings: in our experiment, nodes are PDF objects in a graph and node embeddings are numerical vector representations of each node.
- Classifier: The machine learning model that predict if a PDF is malicious or not.
- Docker Image: built version of the Dockerfile that contains the operating system, Python version, system libraries, python packages, etc.
- Docker File: text file that contains instructions for building a Docker image.

### ComputerSecurityProjectGit/:
|**--data/: stores all input and data files.**  
&nbsp;&nbsp;&nbsp;&nbsp;|--raw/: directory for raw PDF files from dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;|--ir/: directory for PDFObj IR files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--orgs/: directory for ORGS created from IR files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--aorgs/: directory for attributed ORGs with node embeddings.  
|**--models/: stores all trained or pretrained machine learning models.**  
&nbsp;&nbsp;&nbsp;&nbsp;|--bert65k/: directory for Pretrained BERT embeddings.  
&nbsp;&nbsp;&nbsp;&nbsp;|--gin/: directory for trained Graph Isomorphism Network (GIN) models.  
|**--output/: Results generated from scripts.**  
&nbsp;&nbsp;&nbsp;&nbsp;|--metrics/: Includes accuracy of predictions, TPR(portion of malicious PDFs &nbsp;&nbsp;&nbsp;&nbsp;correctly detected), &nbsp;&nbsp;&nbsp;&nbsp;TNR(proportion of benign PDFS correctly detected), TRA(how &nbsp;&nbsp;&nbsp;&nbsp;resistant model is to attacks), etc.  
&nbsp;&nbsp;&nbsp;&nbsp;|--plots/: Includes any Graphs, curves, and visualizations if any.  
|**--scripts/: Python scripts.**  
&nbsp;&nbsp;&nbsp;&nbsp;|--poir.py/: Script to converts from PDFs to PDFObj IR.  
&nbsp;&nbsp;&nbsp;&nbsp;|--buildORGs.py/: Script to constructs Object Reference Graphs (ORGs).  
&nbsp;&nbsp;&nbsp;&nbsp;|--embed_aorg.py/: Script to generates node embeddings (PDFObj2Vec).   
&nbsp;&nbsp;&nbsp;&nbsp;|--trainGIN.py/: Script to trains the GIN classifier.  
|**--Dockerfile/: CPU Docker container setup to reproduce experiment.**  
&nbsp;&nbsp;&nbsp;&nbsp;|--PythonDependencies.txt/: List of Python packages needed for the project to run.  
&nbsp;&nbsp;&nbsp;&nbsp;|--entrypoint.sh/: shell script that acts as the main entry point when running the &nbsp;&nbsp;&nbsp;&nbsp;Docker container as it &nbsp;&nbsp;&nbsp;&nbsp;simplifies running project commands inside the container.


### Datasets Source(From Liu's et al, Experiment)
- https://zenodo.org/records/15532394 

### Contributors
- Aron Chen, James Dobbs, Allison, Jason Huang
