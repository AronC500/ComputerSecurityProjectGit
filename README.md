# PDF Malware Analysis Experiment

### Purpose of this Repository  
- This project attempts to reproduce the work of Liu et al., "Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model"


### Software Requirements:
- Python 3.11+ (Preferred as it's the one used in Liu's' et al. experiment and some dependencies may require newer versions)  
- Rest of python dependencies in ComputerSecurityProjectGit/PythonDependencies.txt


### Keyterms you may need to understand this repository:  
- Object Reference Graphs(ORGS):  
- Intermediate Representation(IR):
- BERT embeddings:
- Graph Isomorphism Network(GIN):
- PDFObj IR:
- node embeddings(In our case, PDFObj2Vec):
- Classifier:
- Raw PDF:
- Attributed ORGs:

### ComputerSecurityProjectGit/:
|--data/: stores all input and data files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--raw/: directory for raw PDF files from dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;|--ir/: directory for PDFObj IR files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--orgs/: directory for ORGS created from IR files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--aorgs/: directory for attributed ORGs with node embeddings.  
|--models/: stores all trained or pretrained machine learning models.  
&nbsp;&nbsp;&nbsp;&nbsp;|--bert65k/: directory for Pretrained BERT embeddings.
&nbsp;&nbsp;&nbsp;&nbsp;|--gin/: directory for trained Graph Isomorphism Network (GIN) models.  
|--output/: Results generated from scripts.  
&nbsp;&nbsp;&nbsp;&nbsp;|--metrics/: Includes accuracy of predictions, TPR(portion of malicious PDFs correctly detected), TNR(proportion of benign PDFS correctly detected), TRA(How resistant model is to attacks), etc.  
&nbsp;&nbsp;&nbsp;&nbsp;|--plots/: Includes any Graphs, curves, and visualizations if any. 
|--scripts/: Python scripts.  
&nbsp;&nbsp;&nbsp;&nbsp;|--poir.py/: Script to converts from PDFs to PDFObj IR.  
&nbsp;&nbsp;&nbsp;&nbsp;|--buildORGs.py/: Script to constructs Object Reference Graphs (ORGs).
&nbsp;&nbsp;&nbsp;&nbsp;|--embed_aorg.py/: Script to generates node embeddings (PDFObj2Vec). 
&nbsp;&nbsp;&nbsp;&nbsp;|--trainGIN.py/: Script to trains the GIN classifier.  
|--Dockerfile/: CPU Docker container setup to reproduce experiment.  
&nbsp;&nbsp;&nbsp;&nbsp;|--PythonDependencies.txt/: List of Python packages needed for the project to run. 
&nbsp;&nbsp;&nbsp;&nbsp;|--entrypoint.sh/: shell script that acts as the main entry point when running the Docker container as it simplifies running project commands inside the container.

### Contributors
- Aron Chen, James Dobbs, Allison, Jason Huang
