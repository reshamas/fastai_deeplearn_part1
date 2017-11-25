# Copy Files from Local Computer to AWS Instance
`scp` = secure copy


### General Syntax
`scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'`

### Example
`scp -i "path to .pem file" "file to be copied from local machine" username@amazoninstance: 'destination folder to copy file on remote machine'`
