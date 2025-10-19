# PDF Malware Analysis Experiment
This project attempts to reproduce the work of Liu et al., "Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model"


Object Reference Graphs(ORGS)
Intermediate Representation(IR)

data/: stores all input and data files.
|--raw/: directory for raw PDF files from dataset.
|--ir/: directory for PDFObj IR files.
|--orgs/: ORGS created from IR files.
|--aorgs/: Attributed ORGs with node embeddings.


models/: stores all trained or pretrained machine learning models.      
|--bert65k/: Pretrained BERT embeddings.
|--gin/: Trained Graph Isomorphism Network (GIN) models.

output/: Results generated from scripts.
|--metrics/: Accuracy, TPR, TNR, TRA, etc.
|--plots/: For any Graphs, curves, and visualizations.

scripts/: Python scripts.
|--poir.py/: Converts PDFs to PDFObj IR
|--build_orgs.py/: Constructs Object Reference Graphs (ORGs)
|--embed_aorg.py/: Generates node embeddings (PDFObj2Vec)
|--train_gin.py/: Trains the GIN classifier

Dockerfile/: CPU Docker container setup to reproduce experiment.
|--Dockerfile.gpu/: GPU-enabled Docker container (optional)
|--requirements.txt/: List of Python packages needed for the project
|--entrypoint.sh/:  Script to run any pipeline step easily inside Docker
