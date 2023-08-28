#!/bin/bash

set -ex

# Upgrade Python version
sudo yum -y install openssl-devel bzip2-devel libffi-devel xz-devel sqlite-devel
wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz    && \
tar xzf Python-3.9.9.tgz && cd Python-3.9.9 && \
./configure --enable-optimizations && \
sudo make altinstall

sudo alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 2
sudo alternatives --set python3 /usr/local/bin/python3.9

# Yarn mods for GPU
sudo chmod a+rwx -R /sys/fs/cgroup/cpu,cpuacct
sudo chmod a+rwx -R /sys/fs/cgroup/devices

#Extra Python Libs
sudo python3 -m pip install numpy scipy scikit-learn pandas tsfresh xgboost scikit-learn tqdm joblib ipywidgets s3fs pyarrow urllib3==1.26.6

# Post bootstrap CUDA Toolkit correction
cat <<'_EOF_'> /home/hadoop/secondstage.sh
#!/bin/bash
while true; do
NODEPROVISIONSTATE=` sed -n '/localInstance [{]/,/[}]/{
/nodeProvisionCheckinRecord [{]/,/[}]/ {
/status: / { p }
/[}]/a
}
/[}]/a
}' /emr/instance-controller/lib/info/job-flow-state.txt | awk ' { print $2 }'`

if [ "$NODEPROVISIONSTATE" == "SUCCESSFUL" ]; then
sleep 10;
echo "Running my post provision bootstrap"


if grep isMaster /mnt/var/lib/info/instance.json | grep false;
then
    sudo wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run -P /home/hadoop/
    sudo chmod +x /home/hadoop/cuda_11.6.0_510.39.01_linux.run
    sudo sh /home/hadoop/cuda_11.6.0_510.39.01_linux.run --silent --override --toolkit --samples --toolkitpath=/usr/local/cuda-116 --samplespath=/usr/local/cuda --no-opengl-libs
    sudo ln -s /usr/local/cuda-116 /usr/local/cuda
    sudo rm /home/hadoop/cuda_11.6.0_510.39.01_linux.run
fi

rm /home/hadoop/secondstage.sh
rm /home/hadoop/secondstage.log
exit;
fi

sleep 10;
done
_EOF_

sudo chmod 755 /home/hadoop/secondstage.sh
nohup /home/hadoop/secondstage.sh 2>&1 >> /home/hadoop/secondstage.log &
exit 0