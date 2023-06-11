# TODO(j.swannack): re-write using the gmail api, this will require oauth2


from typing import List
import os
import smtplib
from email.mime.text import MIMEText

_gpt_email = os.environ["GPT_EMAIL"]
_gpt_email_password = os.environ["GPT_EMAIL_PASSWORD"]
_recipient_email = os.environ["RECIPIENT_EMAIL"]


def send_email(
    subject: str,
    body: str,
    sender: str,
    recipients: List[str],
    password: str,
):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")


def main():
    subject = "Test email"
    body = "This is a test email sent from Python"
    sender = _gpt_email
    recipients = [_recipient_email]
    password = _gpt_email_password

    send_email(subject, body, sender, recipients, password)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
