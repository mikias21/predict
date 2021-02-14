import smtplib
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# SENDGRID API_KEY

def send_issue(email, issue):
    message = Mail(
        from_email = email,
        to_emails = "",
        subject = "Prediction Issue",
        html_content = issue
    ) 

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return True
    except Exception as e:
        return False

def send_password_recovery_link(email, userkey):
    message = Mail(
        from_email = "",
        to_emails = email,
        subject = "Change Password For Predict Account",
        html_content = f"Follow the link to Change Account Password <a href='http://localhost:5000/recover?email={email}&userkey={userkey}'>click here</a>"
    ) 

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return True
    except Exception as e:
        return False
