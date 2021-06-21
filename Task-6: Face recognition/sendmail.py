import os 

EMAIL_ADD = os.environ.get('EMAIL_ADDRESS')  
EMAIL_PASS = os.environ.get('EMAIL_PASSWORD')



def mail():

    import smtplib
    import imghdr
    from email.message import EmailMessage

    
    #Sender_Email =  'EMAIL_PASS'
  
    Reciever_Email = "nisarg4843@gmail.com" 

    #Password = EMAIL_ADD
    
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Alert Message." 
    newMessage['From'] = EMAIL_ADD                  
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('WELCOME HOME! LOOKING GOOD BACK THERE...') 

    with open('D:\\MLOps Summer\\Task 6\\faces\\Nisarg\\Face.jpg', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:

        smtp.login(EMAIL_ADD, EMAIL_PASS)              
        smtp.send_message(newMessage) 

    print('MAIL SENT SUCCESSFULLY!')

