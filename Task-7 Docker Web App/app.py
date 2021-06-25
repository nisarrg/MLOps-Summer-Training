#!/usr/bin/python3

import subprocess as sp
import cgi

print("content-type: text/html")
print()

fs = cgi.FieldStorage()
q = fs.getvalue("cmd")

o = sp.getoutput("sudo "+q)
print(o)
