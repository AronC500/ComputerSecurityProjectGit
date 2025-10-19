# PDF Malware Analysis Experiment

### Purpose of this Repository  
- This project attempts to reproduce the work of Liu et al., "Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model"


### Keyterms you may need to understand this repository:  
- Object Reference Graphs(ORGS):  
- Intermediate Representation(IR):


### ComputerSecurityProjectGit/:  
|--data/: stores all input and data files.  
&nbsp;&nbsp;|--raw/: directory for raw PDF files from dataset.  
&nbsp;&nbsp;|--ir/: directory for PDFObj IR files.  
&nbsp;&nbsp;|--orgs/: ORGS created from IR files.  
&nbsp;&nbsp;|--aorgs/: Attributed ORGs with node embeddings.  
&nbsp;&nbsp;|--models/: stores all trained or pretrained machine learning models.  
&nbsp;&nbsp;|--bert65k/: Pretrained BERT embeddings.  
&nbsp;&nbsp;|--gin/: Trained Graph Isomorphism Network (GIN) models.  
|--output/: Results generated from scripts.  
&nbsp;&nbsp;|--metrics/: Accuracy, TPR, TNR, TRA, etc.  
&nbsp;&nbsp;|--plots/: For any Graphs, curves, and visualizations.  
|--scripts/: Python scripts.  
&nbsp;&nbsp;|--poir.py/: Converts PDFs to PDFObj IR  
&nbsp;&nbsp;|--buildORGs.py/: Constructs Object Reference Graphs (ORGs)  
&nbsp;&nbsp;|--embed_aorg.py/: Generates node embeddings (PDFObj2Vec)  
&nbsp;&nbsp;|--trainGIN.py/: Trains the GIN classifier
|--Dockerfile/: CPU Docker container setup to reproduce experiment.  
&nbsp;&nbsp;|--requirements.txt/: List of Python packages needed for the project  
&nbsp;&nbsp;|--entrypoint.sh/: Script to run any pipeline step easily inside Docker
