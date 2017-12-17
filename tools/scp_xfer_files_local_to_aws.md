# Copy Files from Local Computer to AWS Instance
`scp` = secure copy


### General Syntax
`scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'`

### Examples
```bash
% scp -r . ubuntu@107.22.140.44:~/data/camelhorse 
```

```bash
scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'
```

```bash
scp -i .ssh/aws-key-fast-ai.pem 
ubuntu@ec2-35-165-244-148.us-west2.compute.amazonaws.com:~/nbs/Notebooks/Weights/Predictions/test_preds_rms.dat ~/test_preds_rms.dat
```
