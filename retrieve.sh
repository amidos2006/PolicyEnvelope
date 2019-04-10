if [ -d "/var/sftp/uploads/PolicyEnvelope" ]; then

   sudo rsync -av /var/sftp/uploads/PolicyEnvelope /home/jupyter/Notebooks/Chang > /dev/null
   sudo rm -rf /var/sftp/uploads/PolicyEnvelope > /dev/null

fi
