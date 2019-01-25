# GCP Setup notes

## GCP
- https://cloud.google.com
- [Platform:  GCP](https://forums.fast.ai/t/platform-gcp/27375) (Discourse topic)
- [Tutorial](http://course-v3.fast.ai/start_gcp.html) to get started.
- [Complete Guide](https://arunoda.me/blog/ideal-way-to-creare-a-fastai-node) - starting with $0.2/hour


## GCP (Google Cloud Compute)
- fastai instructions for GCP: 
  - http://course-v3.fast.ai/start_gcp.html
- Console:  
  - https://console.cloud.google.com/compute/instances?project=near-earth-comets-f8c3f&folder&organizationId
- 

## Instance
```bash
gcloud --version
```
output:  
```bash
Google Cloud SDK 222.0.0
bq 2.0.36
core 2018.10.19
gsutil 4.34
```

## Create an Instance on GCP
```bash
% export IMAGE_FAMILY="pytorch-1-0-cu92-experimental"
% export ZONE="us-west2-b"
% export INSTANCE_NAME="my-fastai-instance"
% export INSTANCE_TYPE="n1-highmem-8"
% gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
                --accelerator='type=nvidia-tesla-p4,count=1' \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata='install-nvidia-driver=True' \
        --preemptible
Created [https://www.googleapis.com/compute/v1/projects/near-earth-comets-f8c3f/zones/us-west2-b/instances/my-fastai-instance].
NAME                ZONE        MACHINE_TYPE  PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
my-fastai-instance  us-west2-b  n1-highmem-8  true         10.168.0.2   35.235.122.68  RUNNING
% 
```

## Go to GCP Console and see that the instance has been created
- https://console.cloud.google.com/compute/instances?project=near-earth-comets-f8c3f&folder&organizationId
- Note that this will be the page you have to go to later to **STOP YOUR INSTANCE**.

## Connect to GCP Instance
- Once this is done, you can connect to your instance from the terminal by typing:  
Example:  
```bash
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080
```
For me, it is:  
```bash
gcloud compute ssh --zone=$ZONE jupyter@my-fastai-instance -- -L 8080:localhost:8080
```
###
My passphrase:  fastai

### 
>my example  

```bash
% gcloud compute ssh --zone=$ZONE jupyter@my-fastai-instance -- -L 8080:localhost:8080 
Updating project ssh metadata...⠧Updated [https://www.googleapis.com/compute/v1/projects/near-earth-comets-f8c3f].
Updating project ssh metadata...done.                                                                        
Waiting for SSH key to propagate.
Warning: Permanently added 'compute.7610414667562550937' (ECDSA) to the list of known hosts.
Enter passphrase for key '/Users/reshamashaikh/.ssh/google_compute_engine': 
Enter passphrase for key '/Users/reshamashaikh/.ssh/google_compute_engine': 
======================================
Welcome to the Google Deep Learning VM
======================================

Version: m10
Based on: Debian GNU/Linux 9.5 (stretch) (GNU/Linux 4.9.0-8-amd64 x86_64\n)

Resources:
 * Google Deep Learning Platform StackOverflow: https://stackoverflow.com/questions/tagged/google-dl-platform
 * Google Cloud Documentation: https://cloud.google.com/deep-learning-vm
 * Google Group: https://groups.google.com/forum/#!forum/google-dl-platform

To reinstall Nvidia driver (if needed) run:
sudo /opt/deeplearning/install-driver.sh
This image uses python 3.6 from the Anaconda. Anaconda is installed to:
/opt/anaconda3/

If anything need to be installed and used with Jupyter Lab please do it in the following way:
sudo /opt/anaconda3/bin/pip install <PACKAGE>

Linux my-fastai-instance 4.9.0-8-amd64 #1 SMP Debian 4.9.110-3+deb9u6 (2018-10-08) x86_64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
jupyter@my-fastai-instance:~$ 
```

### Commands to run

```bash
ls
python -V
conda --version
pip list
```

### Go to localhost (run Jupyter Notebook)
http://localhost:8080/tree

    
    
