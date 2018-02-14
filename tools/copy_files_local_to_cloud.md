# Copy Files from Local Computer to Cloud Computer
- copy files from local computer to AWS, Paperspace, Google Cloud, etc
- copy files from cloud computer to local
- copy files from local computer to remote machine

## Reference
[fastai Forum thread](http://forums.fast.ai/t/lesson-1-part-1-v2-custom-images/10154/16)
- [Stack Overflow](https://stackoverflow.com/questions/4728752/scp-a-bunch-of-files-via-bash-script-there-must-be-a-better-way)
- [Stack Exchange](https://unix.stackexchange.com/questions/232946/how-to-copy-all-files-from-a-directory-to-a-remote-directory-using-scp)

## Defintion
`scp` = secure copy

### General Syntax
`scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'`

### Examples
```bash
scp -r . ubuntu@107.22.140.44:~/data/camelhorse 
```

```bash
scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'
```

```bash
scp -i .ssh/aws-key-fast-ai.pem 
ubuntu@ec2-35-165-244-148.us-west2.compute.amazonaws.com:~/nbs/Notebooks/Weights/Predictions/test_preds_rms.dat ~/test_preds_rms.dat
```
