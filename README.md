## Overview
This project demonstrates an end-to-end data science workflow for classifying Iris flower species using a Dockerized deep learning pipeline. The solution combines data scraping, preprocessing, model training, and batch inference within isolated Docker containers for reproducibility and scalability.


## Prerequisites

- **Data Acquisition**: Scrapes Iris dataset from [Scikit-learn](https://scikit-learn.org/)
- **Data Processing**: Splits data into training/inference sets (70/30) with stratified sampling
- **Deep Learning Model**: PyTorch-based neural network classifier
- **Containerized Workflow**:
  - 🐳 **Training Container**: Isolated model training environment
  - 🐳 **Inference Container**: prediction system


### Setting Up Development Environment
Next, you need to set up a suitable Integrated Development Environment (IDE). Visual Studio Code (VSCode) is a great tool for this. You can download it from the official website (https://code.visualstudio.com/Download). After installing VSCode, open it and navigate to the `File` menu and click `Add Folder to Workspace`. Navigate to the directory where you cloned the forked repository and add it. VSCode supports a wide range of programming languages with features like syntax highlighting, code completion, and debugging configurations. You can now edit the files, navigate through your project, and start contributing to `ML_basic_example`. For running scripts, open a new terminal in VSCode by selecting `Terminal -> New Terminal`. Now you can execute your Python scripts directly in the terminal.

### Installing Docker Desktop

Installing Docker Desktop is a straightforward process. Head over to the Docker official website's download page ([Docker Download Page](https://www.docker.com/products/docker-desktop)), and select the version for your operating system - Docker Desktop is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. 

Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. It will typically show up in your applications or programs list. After launching, Docker Desktop will be idle until you run Docker commands. This application effectively wraps the Docker command line and simplifies many operations for you, making it easier to manage containers, images, and networks directly from your desktop. 

Keep in mind that Docker requires you to have virtualization enabled in your system's BIOS settings. If you encounter issues, please verify your virtualization settings, or refer to Docker's installation troubleshooting guide. Now you're prepared to work with Dockerized applications!

### Installing MLFlow on Windows

MLFlow can be easily installed on a Windows local machine using the pip, the Python package installer. To do so, open the command prompt (you can find it by searching for `cmd` in the Start menu) and type the following command:

```python
pip install mlflow
```

After the successful installation, you can start managing and deploying your ML models with MLFlow. For further information on how to use MLFlow at its best, refer to the official MLFlow documentation or use the `mlflow --help` command.

Should you encounter any issues during the installation, you can bypass them by commenting out the corresponding lines in the `train.py` and `requirements.txt` files.

To run MLFlow, type `mlflow ui` in your terminal and press enter. If it doesn't work, you may also try `python -m mlflow ui`  This will start the MLFlow tracking UI, typically running on your localhost at port 5000. You can then access the tracking UI by opening your web browser and navigating to `http://localhost:5000`.


## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── xor_inference_data.csv
│   └── xor_train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:
Data is the cornerstone of any Machine Learning project. For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.

## Acknowledgments
Scikit-learn development team
Alexander Markorvin [https://github.com/MarkKorvin/MLE_basic_example]
PyTorch maintainers
