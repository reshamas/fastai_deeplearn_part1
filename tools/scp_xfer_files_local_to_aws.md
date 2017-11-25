# Copy Files from Local Computer to AWS Instance
tag:  scp  
`scp` = secure copy


## Syntax

### General Syntax
`scp -i "path to .pem file" "file to be copeide from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'`

### Example
`scp -i "path to .pem file" "file to be copeide from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'
