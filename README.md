# SOCIB Shoreline extraction

[![Build Status](https://jenkins.cloud.ai4eosc.eu/buildStatus/icon?job=AI4OS-hub/socib-shoreline-extraction/main)](https://jenkins.cloud.ai4eosc.eu/job/AI4OS-hub/job/socib-shoreline-extraction/job/main/)

Developed by [SOCIB](https://www.socib.es/), this AI module automatically delineates the shoreline in beach imagery. It achieves this through a two-step process: first, a Convolutional Neural Network performs image segmentation, and then an automated post-processing step defines the shoreline as the boundary between the segmented 'landward' and 'seaward' regions. The module integrates with the [DEEPaaS API](https://github.com/ai4os/DEEPaaS), which provides platform support and enhances the functionality and accessibility of the code, allowing users to interact with the detection pipeline efficiently.

![shoreline_extraction_output_example](https://raw.githubusercontent.com/ai4os-hub/socib-shoreline-extraction/main/figures/shoreline_extraction_output_example.png)

The underlying model (DeepLabV3) was trained on labelled oblique and rectified images from the [Spanish CoastSnap Network](https://doi.org/10.1016/j.ocecoaman.2024.107280), and delivers one different solution for each image type. 

## 🚀 Running the container

### ☁️ Directly from Docker Hub

To run the Docker container directly from Docker Hub and start using the API, simply run the following command:


```bash
docker run -ti -p 5000:5000 ai4oshub/socib-shoreline-extraction
```

This command will pull the Docker container from the Docker Hub [ai4oshub](https://hub.docker.com/u/ai4oshub/) repository and start the default command (`deepaas-run --listen-ip=0.0.0.0 --config-file deepaas.conf`).

### 🛠️ Building the container

To build the container directly on your machine (for example, if you need to modify the `Dockerfile`), use the instructions below:
```bash
git clone https://github.com/ai4os-hub/socib-shoreline-extraction
cd socib-shoreline-extraction
docker build -t ai4oshub/socib-shoreline-extraction .
docker run -ti -p 5000:5000 ai4oshub/socib-shoreline-extraction
```

These three steps will download the repository from GitHub and will build the Docker container locally on your machine. You can inspect and modify the `Dockerfile` in order to check what is going on. For instance, you can pass the `--debug=True` flag to the `deepaas-run` command, in order to enable the debug mode.

## 🔌 Connect to the API

Once the container is up and running, browse to http://0.0.0.0:5000/ui to get the [OpenAPI (Swagger)](https://www.openapis.org/) documentation page.

## 📂 Project structure

```
├── .gitignore                     <- List of files ignored by git
├── .sqa/                          <- CI/CD configuration files
│   ├── config.yml                 <- SQA configuration file
│   └── docker-compose.yml         <- Docker compose for SQA testing
├── ai4-metadata.yml               <- Defines information propagated to the AI4OS Hub
├── data/                          <- Example images for testing
│   ├── oblique.jpg
│   └── rectified.jpg
├── deepaas.conf                   <- Configuration file for DEEPaaS API server
├── Dockerfile                     <- Describes steps to build the Docker image
├── JenkinsConstants.groovy        <- Global constants for Jenkins pipeline
├── Jenkinsfile                    <- Describes Jenkins CI/CD pipeline
├── LICENSE                        <- License file
├── models/                        <- Folder to store trained ML models
├── pyproject.toml                 <- Build system dependencies and configuration
├── README.md                      <- README for developers and users
├── requirements.txt               <- List of Python dependencies
├── socib_shoreline_extraction/    <- Main Python package source code
│   ├── api.py                     <- API entry points and definition
│   ├── app/                       <- Core application logic
│   │   ├── data_processing/       <- Scripts for image manipulation
│   │   ├── model/                 <- Neural network architectures
│   │   └── predictor.py           <- Logic for running predictions
│   ├── config.py                  <- Internal application configuration
│   └── schemas.py                 <- API schemas
├── tests/                         <- Unit and integration tests
├── tox.ini                        <- Configuration for tox automation
└── VERSION                        <- Current version of the application
```
# 🇪🇺 Acknowledgements

This work was supported by ‘iMagine’ (Grant Agreement No.101058625) and ‘FOCCUS’ (Grant Agreement No.101133911) European Union funded projects. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA).

## 📚 References
- Soriano-González, J., et al. (2025). Machine learning-driven shoreline extraction and beach seagrass wrack detection from beach imaging systems. In Proceedings of the 10th Coastal Dynamics Conference (Aveiro).
- [Soriano-González, J., et al. (2024). From a citizen science programme to a coastline monitoring system: Achievements and lessons learnt from the Spanish CoastSnap network](https://doi.org/10.1016/j.ocecoaman.2024.107280)
- [Soriano-González, J., et al. (2023). SCLabels Dataset: Labelled rectified RGB images from the Spanish CoastSnap network](https://doi.org/10.5281/zenodo.10159977)
