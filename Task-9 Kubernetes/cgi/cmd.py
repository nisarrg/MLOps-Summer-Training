#!/usr/bin/python3
import cgi
import subprocess

print("content-type: text/html")
print()

f = cgi.FieldStorage()
cmd = f.getvalue('n')
ip="3.109.1.123"
op = subprocess.getoutput("sudo ssh ec2-user@{} -i keyhadoop.pem -o stricthostkeychecking=no  \"sudo {}\"".format(ip,cmd))
print(op)

