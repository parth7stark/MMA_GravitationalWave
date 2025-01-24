# Multimessenger App - Gravitational Wave Data

## Overview
This repository is part of a **Multimessenger App** designed to analyze data from different astronomical sources. In multimessenger astronomy, signals from various messengers—such as gravitational waves, radio waves, and electromagnetic waves—are combined to gain a more comprehensive understanding of astrophysical phenomena.

This repo specifically contains files and code related to **gravitational wave data** analysis, forming one of the core components of the overall multimessenger workflow. The app integrates these data streams with other repositories handling gravitational wave analysis to create a unified event detection and analysis framework.

## Installation and running your first inference

You will need a machine running Linux, MMA_GW does not support other
operating systems.

Please follow these steps:

1.  Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/parth7stark/MMA_GravitationalWave.git
    cd ./MMA_GravitationalWave
    ```

2.  Download inference dataset and model parameters:

    *   Please use the below link to download and set
        up full database. 

    ```bash
    https://drive.google.com/file/d/1zuNzPzHGlk0e5cUCDDPkZjZ5v81J-rhS/view?usp=drive_link
    Unzip the file in <DATASET_DIR>
    ```

    *   **Note: The dataset directory `<DATASET_DIR>` should *not* be a
        subdirectory in the MMA_GravitationalWave repository directory.** If it is, the
        Apptainer build will be slow as the large databases will be copied into the
        Apptainer build context.


3. Build the Apptainer image:

    ```bash
    apptainer build MMA_GW_Inference_miniapp.sif apptainer/MMA_GW_Inference_miniapp.def
    ```

4. Setup Octopus Connection

    To allow the server and detectors to connect to the Octopus event fabric, you need to export the username (user's OpenID) and password (AWS secret key) in the .bashrc file for all sites where the components will run.

    * **Note:** You must configure the .bashrc file for all sites where the Kafka server, clients, or detectors are running.

    You can append these lines to the end of the .bashrc file using the following command:

    ```bash
    cat <<EOL >> ~/.bashrc
    # Kafka Configuration
    export OCTOPUS_AWS_ACCESS_KEY_ID="AKIA4J4XQFUIIB7Y4FG3"
    export OCTOPUS_AWS_SECRET_ACCESS_KEY="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"
    export OCTOPUS_BOOTSTRAP_SERVERS='b-1-public.diaspora.fy49oq.c9.kafka.us-east-1.amazonaws.com:9198,b-2-public.diaspora.fy49oq.c9.kafka.us-east-1.amazonaws.com:9198'
    EOL   
    ```

    After updating the .bashrc, reload it to apply the changes to your current shell session:
    
    ```bash
    source ~/.bashrc
    ```

5. Start Server for Inference


   * Update the config file:

        Open the `examples/configs/FLserver.yaml` configuration file and update the following paths with the appropriate ones for your system:

        - **Checkpoint Path:** Specify the location of the downloaded model checkpoint file.

        - **Logging Output Path:** Define where the inference logs should be saved.


   * Update the Job script:
      
       The sample job script can be located in the repository under the name `job_scripts/FL_Server.sh`

        ```bash
        job_scripts/FL_Server.sh
    
        - Modify the SLURM parameters in the script to suit your computing environment (e.g., partition, time, and resources).
        ```

    * Submit the Job Script
    
        Use the following command to submit the job script:
    
        ```bash
        sbatch job_scripts/FL_Server.sh
        ```

    Submitting the job script will automatically start the Server.

6. Start Detectors for Inference

   Once Server has started running, start both detectors.

   * Update the config file

        Open the `examples/configs/detector0.yaml` and  `examples/configs/detector1.yaml`configuration file and update the following paths with the appropriate ones for your system:

        - **Checkpoint Path**: Specify the location of the downloaded model checkpoint file.
        - **Inference Dataset Path**: Provide the path to the downloaded inference dataset file.
        - **Logging Output Path**: Define where the inference logs should be saved.
    

   * Update the Job script 
   
       The sample job script can be located in the repository under the name `job_scripts/FL_DetectorX.sh`
    
        ```bash
        job_scripts/FL_DetectorX.sh
    
        - Modify the SLURM parameters in the script to suit your computing environment (e.g., partition, time, and resources).
        ```

   * Submit the Job Script
        Use the following command to submit the job script:
    
        ```bash
        sbatch job_scripts/FL_DetectorX.sh
        ```

    Submitting the job script will automatically start the Detectors.


7.  Once the run begins, two output files will be generated for each client and server in your working directory: 
`<job-name>-err-<job_id>.log` (error logs) and `<job-name>-out-<job_id>.log` (output logs). Additionally, the job's output files will be saved in the log output directory specified in your configuration file.


## Todo List and Project Plan
Please refer to our Box folder for the latest project tasks and roadmap: [Link](https://www.overleaf.com/project/66bce960bfb79d8b86fcfdf3)

## Related Projects
This repo focuses on radio wave data. For gravitational wave analysis, please visit [Radio Wave Analysis Repo](https://github.com/parth7stark/MMA_RadioWave/tree/main). Together, these repositories work within the multimessenger framework to capture and analyze various cosmic events.

## Future Plans
- Integration of additional messenger types (e.g., neutrinos, gamma rays)
- Real-time data streaming and event detection
- Cross-correlation between different datasets for enhanced analysis
