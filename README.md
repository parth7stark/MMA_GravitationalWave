# Multimessenger App - Gravitational Wave Data

## Overview
This repository is part of a **Multimessenger App** designed to analyze data from different astronomical sources. In multimessenger astronomy, signals from various messengers—such as gravitational waves, radio waves, and electromagnetic waves—are combined to gain a more comprehensive understanding of astrophysical phenomena.

This repo specifically contains files and code related to **gravitational wave data** analysis, forming one of the core components of the overall multimessenger workflow. The app integrates these data streams with other repositories handling gravitational wave analysis to create a unified event detection and analysis framework.

## Installation and running your first inference

You will need a machine running Linux x86 architecture to build container image. If using ARM64 or other architecture, one need to update Apptainer definition file accordingly

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
    Unzip the file in <REPO_DIR>
    ```

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

## Usage on Delta AI

Since Delta AI is an **ARM64 architecture machine**, so the Apptainer definition file must be updated to use a base image and install libraries compatible with the ARM64 architecture.

Just use the `MMA_GW_Inference_miniapp_arm64.def` for building the image.
The remaining steps are the same as those for Delta or Linux x86 architecture.


## Usage on Polaris

Polaris compute nodes do not have network access to the event fabric's port. Therefore, an HTTPS proxy is required. While this will be set up in the future, as a workaround, you can run the server or detector on the **login node**.

Modification to above usage steps:

**Step 1:** Clone repo (no change)

**Step 2:** Download dataset and checkpoints (no change)

**Step 3:** Build the Apptainer image:

To build the Apptainer images, you need to be on the Polaris compute nodes. Follow these steps:

1. **Start an Interactive Session:**
   ```bash
   qsub -I -A <Project> -q debug -l select=1 -l walltime=01:00:00 -l filesystems=home:eagle -l singularity_fakeroot=true
   ```

2. **Set Proxy Environment Variables:**

   ```bash
   export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
   export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
   export https_proxy=http://proxy.alcf.anl.gov:3128
   ```

3. **Load Apptainer Module:**

   ```bash
   ml use /soft/modulefiles
   ml load spack-pe-base/0.8.1
   ml load apptainer
   ml load e2fsprogs
   ```

4. **Build Apptainer Images:**
   
    ```bash
    apptainer build --fakeroot MMA_GW_Inference_miniapp.sif apptainer/MMA_GW_Inference_miniapp.def
    ```


**Step 4:** Update configurations

Update the configurations for the server and detectors as described in the previous steps. Ensure the paths in the config files point to the appropriate locations on your system.

**Step 5:** Start container on login node

To start the server or detectors, execute the following commands from the login node.

    * To start server, run:

    ```bash
    apptainer exec --nv \
    MMA_GW_Inference_miniapp.sif \
    python /app/examples/octopus/run_server.py --config <absolute path to FL server config file>/FLserver.yaml
    ```

    * To start Detector X, run:

    ```bash
    apptainer exec --nv \
    MMA_GW_Inference_miniapp.sif \
    python /app/examples/octopus/run_detector.py --config <absolute path to FL detectorX config file>/detectorX.yaml
    ```
    

## Todo List and Project Plan
Please refer to our Box folder for the latest project tasks and roadmap: [Link](https://www.overleaf.com/project/66bce960bfb79d8b86fcfdf3)

## Related Projects
This repo focuses on radio wave data. For gravitational wave analysis, please visit [Radio Wave Analysis Repo](https://github.com/parth7stark/MMA_RadioWave/tree/main). Together, these repositories work within the multimessenger framework to capture and analyze various cosmic events.

## Future Plans
- Integration of additional messenger types (e.g., neutrinos, gamma rays)
- Real-time data streaming and event detection
- Cross-correlation between different datasets for enhanced analysis
