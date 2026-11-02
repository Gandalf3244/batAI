"""
Email sender for bat activity reports.
Adapted from gui_email.py for embedded Raspberry Pi system.
"""

import smtplib
import json
import html
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import email
import email.utils
import logging
import queue
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Food consumption prediction (from gui_email.py)
TRENDLINE_SLOPE = 6.4899
TRENDLINE_INTERCEPT = 40.2899


def remove_files(paths: Optional[List[str]]):
    """Delete report graph files (they're embedded in the email once it's built)."""
    for path in paths or []:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Could not delete {path}: {e}")


def predict_food_consumption(want_food_calls_per_hour: float,
                            slope: float = TRENDLINE_SLOPE,
                            intercept: float = TRENDLINE_INTERCEPT) -> float:
    """
    Predict food consumption for next day based on Want_Food calls.
    
    Formula: Food Consumption = slope × (calls/hr) + intercept
    
    Args:
        want_food_calls_per_hour: Want_Food calls per hour
        slope: Trendline slope
        intercept: Trendline intercept
        
    Returns:
        Predicted food consumption
    """
    return slope * want_food_calls_per_hour + intercept


class EmailSender:
    """
    Send bat activity report emails with graphs and predictions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize email sender.
        
        Args:
            config: Email configuration dict with smtp_host, smtp_port, sender_email, etc.
        """
        self.update_config(config)
        logger.info(f"Email sender initialized: {len(self.recipients)} recipient(s)")

    def update_config(self, config: Dict):
        """Apply new email settings (called when settings are edited at runtime)."""
        self.config = config
        self.smtp_host = config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = int(config.get('smtp_port', 587))
        self.sender_email = config.get('sender_email', '')
        self.sender_password = config.get('sender_password', '')
        self.recipients = config.get('recipients', [])

        # Ensure recipients is a list
        if isinstance(self.recipients, str):
            self.recipients = [r.strip() for r in self.recipients.split(',') if r.strip()]

        self.trendline_slope = float(config.get('trendline_slope', TRENDLINE_SLOPE))
        self.trendline_intercept = float(config.get('trendline_intercept', TRENDLINE_INTERCEPT))

    def _connect(self, timeout: float = 30) -> smtplib.SMTP:
        """Open an authenticated SMTP connection (SSL on port 465, STARTTLS otherwise)."""
        if not self.sender_email or not self.sender_password:
            raise ValueError("Sender email or password not configured")
        if self.smtp_port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=timeout)
        else:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=timeout)
            server.ehlo()
            server.starttls()
            server.ehlo()
        server.login(self.sender_email, self.sender_password)
        return server
    
    def send_activity_report(self, recording_date: str, activity_tracker,
                           graph_paths: Optional[List[str]] = None,
                           summary_text: Optional[str] = None,
                           audio_filename: Optional[str] = None) -> str:
        """
        Build and send the activity report email (graph files are deleted after sending).

        Returns:
            Empty string on success, error message on failure
        """
        if not self.recipients:
            return "No recipient email addresses configured."
        try:
            msg = self.build_activity_report(recording_date, activity_tracker, graph_paths,
                                             summary_text, audio_filename)
        except Exception as e:
            error_msg = f"Failed to build report: {e}"
            logger.error(error_msg)
            return error_msg
        error = self.send_raw(msg.as_string())
        if not error:
            logger.info(f"Activity report sent to {len(self.recipients)} recipient(s)")
            remove_files(graph_paths)
        return error

    def build_activity_report(self, recording_date: str, activity_tracker,
                              graph_paths: Optional[List[str]] = None,
                              summary_text: Optional[str] = None,
                              audio_filename: Optional[str] = None) -> MIMEMultipart:
        """
        Build the complete report email (graphs embedded), ready to send now or later.

        Args:
            recording_date: Recording date string (MM/DD/YYYY)
            activity_tracker: ActivityTracker instance with data
            graph_paths: List of graph image paths to attach
            summary_text: Optional summary text
            audio_filename: Optional audio filename for tracking
        """
        # Get duration information
        duration_timedelta = activity_tracker.get_recording_duration()
        duration_seconds = duration_timedelta.total_seconds()
        duration_hours = duration_seconds / 3600
        duration_minutes = duration_seconds / 60
        
        # Get Want_Food rate for Straws (default species for prediction)
        want_food_rate = activity_tracker.get_want_food_rate("Straws")
        predicted_consumption = predict_food_consumption(
            want_food_rate,
            self.trendline_slope,
            self.trendline_intercept
        )
        
        # Calculate next day
        next_day = ""
        try:
            dt = datetime.strptime(recording_date, "%m/%d/%Y")
            next_day = (dt + timedelta(days=1)).strftime("%m/%d/%Y")
        except Exception:
            next_day = "the next day"
        
        # Build subject
        subject = f"Bat Activity Report & Food Prediction — {recording_date}"
        
        # Build plain text body
        body_lines = [
            "Bat Activity Monitoring System — Automated Report",
            "=" * 58,
            "",
            f"Recording date       : {recording_date}",
            f"Prediction for       : {next_day}",
        ]
        
        if audio_filename:
            body_lines.append(f"Audio filename       : {audio_filename}")
        
        body_lines.extend([
            f"Total vocalizations  : {activity_tracker.get_total_vocalizations()}",
            f"Recording duration   : {duration_hours:.1f} hours ({duration_minutes:.1f} minutes)",
            "",
            "─" * 58,
            "FOOD CONSUMPTION PREDICTION",
            "─" * 58,
            "",
            f"Straws Want_Food calls/hour  : {want_food_rate:.2f}",
            "",
            "Trendline used:",
            f"  Food Consumption = {self.trendline_slope} × (calls/hr) + {self.trendline_intercept}",
            "",
            f"Predicted food consumption   : {predicted_consumption:.2f}",
            "",
        ])
        
        # Get species behavior data
        species_behaviors = activity_tracker.get_species_behavior_counts()
        species_counts = activity_tracker.get_species_counts()
        
        # Define expected behaviors for the current four-class model
        rods_behaviors_list = ["Fighting"]
        straws_behaviors_list = ["Fighting", "Want_Food", "Talking"]
        
        # Add detailed overnight activity summary (matching gui_email.py spreadsheet format)
        body_lines.append("─" * 58)
        body_lines.append("OVERNIGHT ACTIVITY SUMMARY (DETAILED DATA)")
        body_lines.append("─" * 58)
        body_lines.append("")
        
        # Rods detailed breakdown
        body_lines.append("Rods Behaviors:")
        body_lines.append("")
        rods_total = species_counts.get("Rods", 0)
        rods_rate = rods_total / max(0.01, duration_hours)
        
        for behavior in rods_behaviors_list:
            count = species_behaviors.get("Rods", {}).get(behavior, 0)
            rate = count / max(0.01, duration_hours)
            body_lines.append(f"  {behavior:20s}: {count:5d} total  |  {rate:6.1f} calls/hr")
        
        body_lines.append(f"  {'TOTAL':20s}: {rods_total:5d} total  |  {rods_rate:6.1f} calls/hr")
        body_lines.append("")
        
        # Straws detailed breakdown
        body_lines.append("Straws Behaviors:")
        body_lines.append("")
        straws_total = species_counts.get("Straws", 0)
        straws_rate = straws_total / max(0.01, duration_hours)
        
        for behavior in straws_behaviors_list:
            count = species_behaviors.get("Straws", {}).get(behavior, 0)
            rate = count / max(0.01, duration_hours)
            body_lines.append(f"  {behavior:20s}: {count:5d} total  |  {rate:6.1f} calls/hr")
        
        body_lines.append(f"  {'TOTAL':20s}: {straws_total:5d} total  |  {straws_rate:6.1f} calls/hr")
        body_lines.append("")
        
        # Combined totals
        combined_total = rods_total + straws_total
        combined_rate = combined_total / max(0.01, duration_hours)
        
        body_lines.append("Combined (Both Species):")
        body_lines.append(f"  Total vocalizations  : {combined_total}")
        body_lines.append(f"  Calls per hour       : {combined_rate:.1f}")
        body_lines.append("")
        
        # Duration summary
        body_lines.append("Recording Duration:")
        body_lines.append(f"  {duration_minutes:.1f} minutes")
        body_lines.append(f"  {duration_hours:.1f} hours")
        body_lines.append("")
        
        # Add species percentage breakdown
        body_lines.append("─" * 58)
        body_lines.append("SPECIES BREAKDOWN (PERCENTAGES)")
        body_lines.append("─" * 58)
        body_lines.append("")
        
        total = activity_tracker.get_total_vocalizations()
        for species, count in sorted(species_counts.items()):
            percentage = (count / max(1, total)) * 100
            body_lines.append(f"{species:20s}: {count:5d} ({percentage:5.1f}%)")
        
        body_lines.append("")
        
        # Add behavior percentage breakdown per species
        body_lines.append("─" * 58)
        body_lines.append("BEHAVIOR BREAKDOWN (PERCENTAGES)")
        body_lines.append("─" * 58)
        body_lines.append("")
        
        for species in sorted(species_behaviors.keys()):
            behaviors = species_behaviors[species]
            species_total = sum(behaviors.values())
            body_lines.append(f"{species}:")
            for behavior, count in sorted(behaviors.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / max(1, species_total)) * 100
                body_lines.append(f"  {behavior:20s}: {count:5d} ({percentage:5.1f}%)")
            body_lines.append("")
        
        # Add custom summary if provided
        if summary_text:
            body_lines.append("─" * 58)
            body_lines.append(summary_text)
            body_lines.append("")
        
        body_lines.extend([
            "─" * 58,
            "Generated automatically by Bat Activity Monitoring System",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        body = "\n".join(body_lines)
        
        # Build HTML body with detailed data table
        html_lines = [
            "<html><body style='font-family: Arial, sans-serif;'>",
            "<pre style='font-family: monospace; background-color: #f5f5f5; padding: 15px; border-radius: 5px;'>",
            html.escape(body),
            "</pre>",
        ]
        
        # Add detailed data table (matching gui_email.py spreadsheet format)
        html_lines.append("<br><hr>")
        html_lines.append("<h3>Overnight Activity Summary (Detailed Data Table)</h3>")
        html_lines.append("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse: collapse; font-family: Arial, sans-serif;'>")
        html_lines.append("<tr>")
        
        # Build table headers and data
        table_data = {}
        
        # Filename and Date
        if audio_filename:
            table_data['Filename'] = audio_filename
        table_data['Date'] = recording_date
        
        # Rods behaviors
        for behavior in rods_behaviors_list:
            count = species_behaviors.get("Rods", {}).get(behavior, 0)
            rate = count / max(0.01, duration_hours)
            table_data[f'Rods_{behavior}_Total_Calls'] = count
            table_data[f'Rods_{behavior}_Calls_Per_Hour'] = round(rate, 1)
        
        # Straws behaviors
        for behavior in straws_behaviors_list:
            count = species_behaviors.get("Straws", {}).get(behavior, 0)
            rate = count / max(0.01, duration_hours)
            table_data[f'Straws_{behavior}_Total_Calls'] = count
            table_data[f'Straws_{behavior}_Calls_Per_Hour'] = round(rate, 1)
        
        # Species totals
        table_data['Rods_Total_Vocalizations'] = rods_total
        table_data['Rods_Total_Calls_Per_Hour'] = round(rods_rate, 1)
        table_data['Straws_Total_Vocalizations'] = straws_total
        table_data['Straws_Total_Calls_Per_Hour'] = round(straws_rate, 1)
        table_data['Combined_Total_Vocalizations'] = combined_total
        table_data['Combined_Total_Calls_Per_Hour'] = round(combined_rate, 1)
        
        # Duration
        table_data['Total_Duration_Minutes'] = round(duration_minutes, 1)
        table_data['Total_Duration_Hours'] = round(duration_hours, 1)
        
        # Create table headers
        for col_name in table_data.keys():
            html_lines.append(f"<th style='background:#f0f0f0; text-align: left; padding: 6px;'>{html.escape(str(col_name))}</th>")
        
        html_lines.append("</tr><tr>")
        
        # Create table data row
        for col_value in table_data.values():
            html_lines.append(f"<td style='padding: 6px;'>{html.escape(str(col_value))}</td>")
        
        html_lines.append("</tr></table>")
        
        # Add embedded graphs if provided
        if graph_paths:
            html_lines.append("<br><hr>")
            html_lines.append("<h3>Activity Graphs:</h3>")
            for idx, graph_path in enumerate(graph_paths):
                if os.path.exists(graph_path):
                    html_lines.append(f"<img src='cid:graph{idx}' style='max-width:100%; height:auto; margin:10px 0;'><br>")
        
        html_lines.append("</body></html>")
        html_body = "\n".join(html_lines)
        
        # Create message: mixed( related( alternative(text, html), inline images ), attachments )
        msg = MIMEMultipart('mixed')
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = subject

        msg_related = MIMEMultipart('related')
        msg.attach(msg_related)

        # Alternative part for plain text and HTML
        msg_alternative = MIMEMultipart('alternative')
        msg_related.attach(msg_alternative)

        msg_alternative.attach(MIMEText(body, 'plain', 'utf-8'))
        msg_alternative.attach(MIMEText(html_body, 'html', 'utf-8'))
        
        # Attach graph images (both as embedded and attachments)
        if graph_paths:
            for idx, graph_path in enumerate(graph_paths):
                if os.path.exists(graph_path):
                    try:
                        # Embed in HTML
                        with open(graph_path, 'rb') as f:
                            img = MIMEImage(f.read())
                        img.add_header('Content-ID', f'<graph{idx}>')
                        img.add_header('Content-Disposition', 'inline',
                                     filename=os.path.basename(graph_path))
                        msg_related.attach(img)
                        
                        # Also attach as regular attachment
                        with open(graph_path, 'rb') as f:
                            attachment = MIMEImage(f.read())
                        attachment.add_header('Content-Disposition', 'attachment',
                                            filename=os.path.basename(graph_path))
                        msg.attach(attachment)
                        
                    except Exception as e:
                        logger.warning(f"Could not attach graph {graph_path}: {e}")
        
        return msg

    def send_raw(self, raw: str, recipients: Optional[List[str]] = None) -> str:
        """
        Send an already-built message. Recipients default to the message's To: header
        (so a report saved while offline still goes to the people it was written for).

        Returns:
            Empty string on success, error message on failure
        """
        if recipients is None:
            to = email.message_from_string(raw).get_all("To", [])
            recipients = [addr for _name, addr in email.utils.getaddresses(to) if addr] or self.recipients
        if not recipients:
            return "No recipient email addresses configured."
        try:
            with self._connect() as server:
                server.sendmail(self.sender_email, recipients, raw)
            return ""
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def test_connection(self) -> str:
        """
        Test email connection and credentials.
        
        Returns:
            Empty string on success, error message on failure
        """
        try:
            with self._connect(timeout=15):
                pass

            logger.info("Email connection test successful")
            return ""

        except Exception as e:
            error_msg = f"Email connection failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def send_test_email(self) -> str:
        """
        Send a short test email to all recipients.

        Returns:
            Empty string on success, error message on failure
        """
        if not self.recipients:
            return "No recipient email addresses configured."
        try:
            msg = MIMEText(
                "This is a test email from the Bat Activity Monitoring System.\n\n"
                "If you received it, nightly reports will be delivered to this address.\n\n"
                f"Sent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'plain', 'utf-8')
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = "Bat Monitor — test email"
            with self._connect(timeout=15) as server:
                server.sendmail(self.sender_email, self.recipients, msg.as_string())
            logger.info(f"Test email sent to {len(self.recipients)} recipient(s)")
            return ""
        except Exception as e:
            error_msg = f"Test email failed: {str(e)}"
            logger.error(error_msg)
            return error_msg


class EmailQueue:
    """
    Sends reports in the background with retries.

    A report is never thrown away: if it still can't be sent after the retries
    (no WiFi, a network that needs a sign-in page, wrong email password...), or
    the monitor shuts down first, the finished email is saved in the outbox
    folder and re-sent every few minutes - and after a restart - until it goes.
    """

    def __init__(self, email_sender: EmailSender, max_retries: int = 3,
                 retry_delay: int = 60, outbox_dir: Optional[str] = None,
                 outbox_interval: float = 600):
        """
        Args:
            email_sender: EmailSender instance
            max_retries: Attempts before a report moves to the outbox
            retry_delay: Delay between attempts in seconds
            outbox_dir: Folder for reports waiting to be sent (None = keep only in memory)
            outbox_interval: Seconds between outbox delivery attempts
        """
        self.email_sender = email_sender
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.outbox_dir = outbox_dir
        self.outbox_interval = outbox_interval

        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self._stop_event = threading.Event()
        self._next_outbox_try = 0.0

        logger.info(f"Email queue initialized (max_retries={max_retries})")

    def enqueue(self, recording_date: str, activity_tracker,
                graph_paths: Optional[List[str]] = None,
                summary_text: Optional[str] = None,
                audio_filename: Optional[str] = None):
        """Add a report to the queue."""
        email_data = {
            'recording_date': recording_date,
            'activity_tracker': activity_tracker,
            'graph_paths': graph_paths,
            'summary_text': summary_text,
            'audio_filename': audio_filename,
            'attempts': 0,
            'raw': None,
        }
        self.queue.put(email_data)
        logger.info("Email added to queue")

    # -- building / outbox --------------------------------------------------

    def _build(self, email_data) -> Optional[str]:
        """Build the email once; afterwards the graphs and tracker aren't needed."""
        if email_data.get('raw') is None:
            try:
                msg = self.email_sender.build_activity_report(
                    email_data['recording_date'], email_data['activity_tracker'],
                    email_data['graph_paths'], email_data['summary_text'], email_data.get('audio_filename'))
                email_data['raw'] = msg.as_string()
            except Exception as e:
                logger.error(f"Could not build the report email: {e}")
                return None
            finally:
                remove_files(email_data.get('graph_paths'))
            email_data['activity_tracker'] = None
        return email_data['raw']

    def _save_to_outbox(self, raw: str):
        if not self.outbox_dir:
            logger.error("Report could not be sent and no outbox is configured - it is lost")
            return
        os.makedirs(self.outbox_dir, exist_ok=True)
        path = os.path.join(self.outbox_dir, f"report_{datetime.now():%Y%m%d_%H%M%S_%f}.eml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
        logger.warning(f"Report saved to the outbox, will keep retrying: {path}")

    def outbox_files(self) -> List[str]:
        if not self.outbox_dir or not os.path.isdir(self.outbox_dir):
            return []
        return sorted(os.path.join(self.outbox_dir, f) for f in os.listdir(self.outbox_dir) if f.endswith(".eml"))

    def flush_outbox(self) -> int:
        """Try to send saved reports, oldest first. Returns how many were sent."""
        sent = 0
        for path in self.outbox_files():
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            if self.email_sender.send_raw(raw):
                break                      # still offline - try again later
            os.remove(path)
            sent += 1
            logger.info(f"Delivered saved report {os.path.basename(path)}")
        return sent

    # -- worker -------------------------------------------------------------

    def _worker(self):
        logger.info("Email queue worker started")
        while self.running:
            try:
                if time.time() >= self._next_outbox_try and self.outbox_files():
                    self._next_outbox_try = time.time() + self.outbox_interval
                    try:
                        self.flush_outbox()
                    except Exception as e:
                        logger.error(f"Outbox delivery error: {e}")

                email_data = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                raw = self._build(email_data)
                if raw is None:
                    continue
                error = self.email_sender.send_raw(raw)
                if not error:
                    logger.info("Report email sent")
                    continue
                email_data['attempts'] += 1
                if email_data['attempts'] < self.max_retries and not self._stop_event.is_set():
                    logger.warning(f"Email send failed (attempt {email_data['attempts']}): {error}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    self._stop_event.wait(self.retry_delay)       # wakes early on shutdown
                    self.queue.put(email_data)
                else:
                    logger.error(f"Email send failed after {email_data['attempts']} attempt(s): {error}")
                    self._save_to_outbox(raw)
            except Exception as e:
                logger.error(f"Error in email worker: {e}")
            finally:
                self.queue.task_done()
        logger.info("Email queue worker stopped")

    def start(self):
        """Start queue worker (saved reports are retried right away)."""
        if not self.running:
            self.running = True
            self._stop_event.clear()
            self._next_outbox_try = 0.0
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            logger.info("Email queue started")

    def stop(self, wait: bool = True, drain_timeout: float = 60.0):
        """
        Stop queue worker. Reports still waiting are tried once more and, if
        that fails, saved to the outbox - never dropped.
        """
        self._stop_event.set()
        if wait and self.running and self.queue.unfinished_tasks:
            logger.info(f"Delivering {self.queue.unfinished_tasks} queued email(s) before stopping...")
            deadline = time.time() + drain_timeout
            while self.queue.unfinished_tasks and time.time() < deadline:
                time.sleep(0.2)
        self.running = False
        if wait and self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        while True:                        # anything the worker didn't get to
            try:
                email_data = self.queue.get_nowait()
            except queue.Empty:
                break
            raw = self._build(email_data)
            if raw:
                self._save_to_outbox(raw)
        logger.info("Email queue stopped")

    def get_queue_size(self) -> int:
        """Reports waiting to be sent (in memory + saved in the outbox)."""
        return self.queue.qsize() + len(self.outbox_files())


def test_email_sender():
    """Test email sender (requires configuration)."""
    print("Testing Email Sender...")
    
    # Load config
    from config_manager import ConfigManager
    config_mgr = ConfigManager()
    email_config = config_mgr.get_email_config()
    
    if not email_config.get('sender_email') or not email_config.get('sender_password'):
        print("✗ Email not configured. Please set email credentials in config.json")
        print("  You can configure email with:")
        print("    config.set_email_config(sender_email='...', sender_password='...', recipients=['...'])")
        return
    
    # Test connection
    sender = EmailSender(email_config)
    error = sender.test_connection()
    
    if error:
        print(f"✗ Email connection test failed: {error}")
    else:
        print("✓ Email connection test successful!")
        
        # Ask if user wants to send test email
        response = input("\nSend test email? (y/n): ")
        if response.lower() == 'y':
            # Create dummy activity tracker
            from activity_tracker import ActivityTracker
            from datetime import datetime
            
            tracker = ActivityTracker()
            # Add some test data
            tracker.add_classification("straws_want_food", 0.95, datetime.now())
            tracker.add_classification("rods_fighting", 0.87, datetime.now())
            
            error = sender.send_activity_report(
                datetime.now().strftime("%m/%d/%Y"),
                tracker,
                summary_text="This is a test email from the Bat Activity Monitoring System"
            )
            
            if error:
                print(f"✗ Test email failed: {error}")
            else:
                print("✓ Test email sent successfully!")


if __name__ == "__main__":
    test_email_sender()
