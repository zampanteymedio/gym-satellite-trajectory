#######################################################################
# Use this script to initialize an EC2 instance ready to use the envs #
#######################################################################

yum check-update
sudo yum update -y

# General tools
sudo yum install java-1.8.0-openjdk-devel -y
sudo yum install gcc-c++ -y
sudo yum install git -y

# Install anaconda
anaconda_file="Anaconda3-2020.02-Linux-x86_64.sh"
wget -q -c "https://repo.continuum.io/archive/${anaconda_file}"
# You might prefer to get the file from a local S3 bucket
# aws s3 cp "s3://awstest2-files/${anaconda_file}" .
chmod +x $"{anaconda_file}"
bash "./${anaconda_file}" -b -f -p ~/anaconda
export PATH="${PATH}:~/anaconda/bin"
rm -f "${anaconda_file}"

# Install Orekit
conda config --add channels conda-forge
conda install -y orekit=10.0

# Install Other python dependencies
pip install PyHamcrest
pip install gym

# Install this package
git clone https://github.com/zampanteymedio/gym-satellite-trajectory.git
cd gym-satellite-trajectory
pip install -e .
cd ..
