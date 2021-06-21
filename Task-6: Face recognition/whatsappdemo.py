from imghdr import what
import pywhatkit
from datetime import datetime

curr_time = datetime.now()
hour = curr_time.hour
min = curr_time.minute

def whatpy():
    number = '#Your Mobile Number'
    text = 'Activity detected. Check you E-mail!'
    pywhatkit.sendwhatmsg(number, text,int(hour), int(min+ 1))
    print('WhatsApp Message sent!')

