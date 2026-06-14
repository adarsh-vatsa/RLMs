## Technical Specifications
Built on reliable hardware and HPC management software from DELL, the latest storage technology Kalray Pixstor, high-speed networking (up to 100Gb HDR) from InfiniBand, familiar scheduler SLURM running in Rocky Linux, it offers researchers access to best-in-class Intel CPU and NVIDIA graphics.

# Nodes
## Technical Details
- 1 Login Node (jarvis.stevens.edu)
- 1 Head Node
- 2nd Head Node (Coming Soon)
- 42 Dual Socket Intel Xeon Platinum  8358 CPU @ 2.60GHz Nodes (2688 cores total)
-  6 Dual Intel(R) Xeon(R) Gold 6444Y @ 3.6GHz Nodes (384 total)
- 2 Dual Socket Intel(R) Xeon(R) Platinum 8462Y+ quad NVIDIA L40S GPU Nodes
- 3 Dual Intel(R) Xeon(R) Platinum 8462Y+ @ 4.1GHz quad NVIDIA L40S GPU Nodes
- 4 Dual Socket Intel(R) Xeon(R) Platinum 8462Y+ dual NVIDIA H100 NVL GPU Nodes  
- 4 Dual Socket Intel(R) Xeon(R) Platinum 8462Y+ quad NVIDIA H100 SXM GPU Nodes 
- 3 Dual Intel(R) Xeon(R) 6747P @ 2.7GHz NVIDIA H200 GPU Nodes
- 3 Dual Socket Intel(R) Xeon(R) Gold 6444Y 4.00 GHz Nodes

| Count | Cores/node | Total Cores | Processor / Graphics                                                                                                     | Mem / Node   | Disks, type, usable | Tag         |
| ----- | ---------- | ----------- | ------------------------------------------------------------------------------------------------------------------------ | ------------ | ------------------- | ----------- |
| 1     |            | -           | Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz                                                                               | 128 GB       | 1x, SSD, 960 GB     | login       |
| 4     | 64         | 256         | Intel Xeon Platinum  8358 CPU @ 2.60GHz                                                                                  | 1.0Ti        | 2x, SAS SSD, 960 GB | bigmem      |
| 38    | 64         | 2432        | Intel Xeon Platinum  8358 CPU @ 2.60GHz                                                                                  | 256 GB       | 2x, SAS SSD, 960 GB | compute     |
| 3     | 32         | 96          | Intel(R) Xeon(R) Gold 6444Y 4.00 GHz                                                                                     | 512 GB       | 892.8 GB            | compute-hf  |
| 6     | 64         | 384         | Intel(R) Xeon(R) Gold 6444Y @ 3.6GHz                                                                                     | 512 GB       | 894.3 GB            | compute-v2  |
| 4     | 64         | 256         | Intel Xeon Platinum  8358 CPU @ 2.60GHz / H100                                                                           | 1 TB         | 2x, SAS SSD, 960 GB | gpu-h100    |
| 2+3   | 32+64      | 256         | Intel Xeon Platinum  8358 CPU @ 2.60GHz / L40s (4) <br><br>+<br><br>Intel(R) Xeon(R) Platinum 8462Y+ @ 4.1GHz / L40s (4) | 256 / 256 GB | 2x, SAS SSD, 960 GB | gpu-l40s    |
| 4     | 32         | 128         | Intel(R) Xeon(R) Platinum 8462Y+                                                                                         | 512 GB       | NVMe SSD, 1.5 TB    | gpu-h100sxm |
| 3     | 96         | 288         | Intel(R) Xeon(R) 6747P @ 2.7GHz                                                                                          | 1.5 TB       | NVMe SSD, 1.6 TB    | gpu-h200    |

# Queue Details
| Name          | Purpose                             | Req Input | Limits/j / u (cores, run time, priority, max#, etc.) | Valid Resources       | Valid Users | Policy |
| ------------- | ----------------------------------- | --------- | ---------------------------------------------------- | --------------------- | ----------- | ------ |
| bigmem        | Jobs that utilize high memory       | -         | Runtime/j: 72 hours                                  | b[001-004]            | All         |        |
| compute       | Standard compute jobs               | -         | Runtime/j: 72 hours  <br>Priority: Medium            | c[010-038]            | All         |        |
| compute-hf    | High frequency compute jobs         | -         | Runtime/j: 72 hours                                  | c[101-103]            | All         |        |
| compute-long  | Long-term compute jobs, one week    | -         | Runtime/j: 168 hours  <br>Priority: Low              | c[010-038]            | All         |        |
| compute-short | Quick compute jobs, one day or less | -         | Runtime/j: 24 hours  <br>Priority: High              | c[010-038]            | All         |        |
| compute-v2    | Standard compute jobs on faster CPU | -         | Runtime/j: 72 hours  <br>Priority: Medium            | b[005-010]            | All         |        |
| davidson-dev  | Reserved for Davidson Development   | -         | Runtime/j: Unlimited  <br>Priority: Highest          | c[007-009]            | Davidson    |        |
| davidson-prod | Reserved for Davidson Production    | -         | Runtime/j: Unlimited  <br>Priority: Highest          | c[001-006]            | Davidson    |        |
| gpu-h100      | H100 GPU Jobs                       | -         | Runtime/j: 24 hours                                  | g[001-004]            | Limited     |        |
| gpu-h200      | H200 GPU Jobs                       | -         | Runtime/j: 24 hours                                  | g[211-213]            | Limited     |        |
| gpu-l40s      | L40S GPU Jobs                       | -         | Runtime/j: 24 hours  <br>Priority: Medium            | g[101-102],g[201-203] | All         |        |
| gpu-h100sxm   | H100 SXM GPU Jobs                   | -         | Runtime/j: 72 hours                                  | g[011-014]            | Limited     |        |

# Storage Details
- Clustered Storage
	- 1.2 PB of SAS/SATA/SSD disks
	- Pixstor GPFS Filesystem - /mmfs1
	- Used for Project Data, /home (/home is a simlink to /mmfs1/home)
- Local Scratch
	- 850 GB of SSD on each compute node
	- /local
- Local Head Node storage
	- 800 GB
	- /cm/shared

# Project-Specific LLM Cache Policy
- Use `jarvis/run.sh` as the project dispatcher for L40S vLLM services, model prefetch jobs, and client jobs.
- Role scripts are split by responsibility: `jarvis/serve_vllm.sh`, `jarvis/run_client.sh`, `jarvis/download_models.sh`, and shared setup in `jarvis/lib/env.sh`.
- See `jarvis/docs/README.md` for model sizes, endpoint wiring, smoke-test mode, serious two-service benchmark mode, and cleanup behavior.
- See `jarvis/docs/HPC_RUNBOOK.md` for the step-by-step Jarvis execution sequence.
- If `/mmfs1/project/llm_caching` is unavailable, use `JARVIS_STORAGE_MODE=scratch`.
- In scratch mode, model assets live under node-local `/local/$USER/llm_caching`, while active vLLM jobs serve from `/local/$USER/$SLURM_JOB_ID/adarsh-rlms`.
- Keep only small logs and endpoint URL files under `/home/edogu/adarsh-rlms-logs`.
- Cleanup should delete only the per-job `/local` runtime directory. Do not automatically delete `/local/$USER/llm_caching`.

# Important: Best Practices
- Use the appropriate partition and resources for your job. Do not request more resources than you need, as this will affect the performance and efficiency of the cluster.
- Do not run jobs on the login node, they should be scheduled via slurm.
- Use the scratch space for temporary files and data. Do not store large or permanent files on the home directory, as this will affect the backup and recovery of the cluster.
- Use the modules system to load and unload the software and libraries that you need for your job. Do not install software on the cluster without permission, as this will affect the security and compatibility of the cluster.
- Use the best practices for parallel programming and optimization. Do not run serial or inefficient code on the cluster, as this will affect the speed and quality of your results.
- Respect the policies and rules of the cluster. Do not abuse or misuse the cluster, as this will affect the availability and reliability of the cluster.

# Submitting Jobs on JARVIS H100 Queue
There are two ways to use a compute node, whether a CPU node or GPU node: interactive (srun) or batch (sbatch).

To request an interactive session on a H100 GPU node using srun, you can use the following command:

`$ srun --partition=gpu-h100 --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:2 --time=01:00:00 --pty bash`

Or if you wanted to allocate half an H100 node (32 cores and 1 GPU), allowing others to share resources:  

`$ srun --partition=gpu-h100 --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --time=01:00:00 --pty bash`

Explanation:  

srun: Runs an interactive job on an allocated node.  
- --partition=gpu-h100: Requests a node from the GPU partition.  
- --nodes=1: Allocates an entire node.  
- --ntasks=1: Runs a single task.  
- --cpus-per-task=64: Requests all 64 CPU cores.  
- --gres=gpu:2: Allocates both H100 GPUs.  
- --time=01:00:00: Limits the session to 1 hour. You can adjust this. 
- --pty bash: Starts an interactive Bash shell on the allocated node.  
 
Once inside the interactive session, you can run commands as needed, such as checking the GPUs with:  

$ module load cuda12.2  # cuda12.0, 12.1, and 12.4 are also available
$ nvidia-smi

or running your program manually:  
$ module list # see if CUDA is already loaded  
$ module load cuda12.2 # if you need to load it. 
$ ./my_gpu_program  

However, you may not always be able to get an interactive session in a timely manner, as other people may be using the resources. What you ought to do is once you have your program ready to go, is to submit a batch job. For that you need to make a batch script, like so:  
  
h100-sample.sh:  
```
#!/bin/bash 

#SBATCH --job-name=my_gpu_job   
#SBATCH --partition=gpu-h100   
#SBATCH --nodes=1   
#SBATCH --ntasks=1   
#SBATCH --cpus-per-task=64   
#SBATCH --gres=gpu:2   
#SBATCH --time=01:00:00 # Set desired wall time. Max runtime is 1 day for the h100 partition, 3 days for h100sxm. 
#SBATCH --output=slurm-%j.out   
#SBATCH --error=slurm-%j.err   
#SBATCH --mail-user=jhong8@stevens.edu   
#SBATCH --mail-type=BEGIN,END,FAIL # Get email notifications for job start, end, and failure  

# Load any required modules  
module load cuda12.2 # Adjust as needed  

# Activate virtual environment (if using one)  
source ~/myenv/bin/activate    

# Run your GPU job  
srun ./my_gpu_program  
```

Once you have your batch script written, you can submit it like so:  

`$ sbatch h100-sample.sh`

Once your job is submitted, it will go into the queue. If there are no other jobs ahead of yours, it will run right away, but if the system is busy (use sinfo to check) and the partition full, it will wait until resources are available, and then run. 

There are currently four GPU partitions on Jarvis that can be used. They are 

gpu-h100, mentioned above. There are four nodes. Each node has 2x32 core CPUs, 256GB of RAM, and 2 H100 GPUs. Max runtime for a job is 24 hours. 
gpu-h100sxm. There are four nodes. Each node has 2x32 core CPUs, 512GB of RAM, and 4 H100 GPUs. Max runtime for a job is 72 hours. 
gpu-l40s. There are two nodes. Each node has 2x32 core CPUs, 256GB of RAM, and 4 L40S GPUs. Max runtime for a job is 24 hours. 
gpu-h200. There are 3 nodes. Each node has 2x48 core Intel(R) Xeon(R) 6747P @ 2.7GHz and 4 H200 GPUs. Max runtime for a job is 24 hours.