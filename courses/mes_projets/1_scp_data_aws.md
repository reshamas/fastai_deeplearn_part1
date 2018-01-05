
```bash
% pwd
/Users/reshamashaikh/ds/data/camelshorses
% ls
total 0
drwxr-xr-x  103   3502 Nov 25 12:19 camels
drwxr-xr-x  104   3536 Nov 25 12:16 horses
% ls camels | wc -l
     102
% ls horses | wc -l
     102
% scp -r . ubuntu@34.198.228.48:~/data/camelhorse 
```
