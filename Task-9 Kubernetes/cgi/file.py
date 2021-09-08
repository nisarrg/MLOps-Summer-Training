#!/usr/bin/python3
import cgi
import subprocess
import os
print("content-type: text/html")
print()

f = cgi.FieldStorage()
ip = '3.109.1.123'
img = f.getvalue('i')
output = subprocess.getoutput("sudo  wget  -O  /var/www/cgi-bin/file.yml {} ".format(img))
output = subprocess.getoutput("sudo scp -i keyhadoop.pem -o stricthostkeychecking=no file.yml ec2-user@{}:/home/ec2-user/file.yml".format(ip))
output = subprocess.getoutput("sudo ssh ec2-user@{} -i keyhadoop.pem -o stricthostkeychecking=no \"sudo kubectl apply -f /home/ec2-user/file.yml\"".format(ip))
o = subprocess.getoutput("sudo ssh ec2-user@{} -i keyhadoop.pem -o stricthostkeychecking=no  \"sudo rm -rf /home/ec2-user/file.yml\"".format(ip))
o = subprocess.getoutput("sudo rm -rf file.yml")
print(output)

