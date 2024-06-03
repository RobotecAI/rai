import logging
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional


class EmailSender:
    def __init__(
        self, smtp_server: str, smtp_port: int, logging_level: int = logging.INFO
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = os.environ.get("ROBOT_ALERT_EMAIL", None)
        self.sender_password = os.environ.get("ROBOT_ALERT_PASSWORD", None)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

        if self.sender_email is None or self.sender_password is None:
            self.logger.error(
                "Email and password for the alert system are not set. Message will not be sent."
            )

    def send_email(
        self,
        recipient_email: str,
        subject: str,
        message: str,
        image_path: Optional[str] = None,
    ) -> None:
        # Create a multipart message
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        # Attach the message as plain text
        msg.attach(MIMEText(message, "html"))
        # Attach the image if provided
        if image_path:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image = MIMEImage(image_data, name="image.png")
            msg.attach(image)

        # Connect to the SMTP server and send the email
        if self.sender_email is None or self.sender_password is None:
            return

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            try:
                server.login(self.sender_email, self.sender_password)
            except smtplib.SMTPAuthenticationError:
                self.logger.error("Failed to authenticate with the SMTP server.")
                return
            server.send_message(msg)
            self.logger.info(f"Email sent to {recipient_email}.")
