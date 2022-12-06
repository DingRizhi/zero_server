source /home/adt/anaconda3/bin/activate lupinus
bash ./stop.sh
cd /home/adt/micro-i/deep_model/zero_server/wlib
nohup python -m libcom_server &
