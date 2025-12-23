"""
nAI Toolpacks - Email Pack
==========================

Email parsing for IMAP, MBOX, and EML files.
"""

import email
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, BinaryIO
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime


@dataclass
class EmailAttachment:
    """An email attachment."""
    filename: str
    content_type: str
    size: int
    content: Optional[bytes] = None


@dataclass
class ParsedEmail:
    """A parsed email message."""
    message_id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str]
    date: Optional[datetime]
    body_text: str
    body_html: Optional[str]
    attachments: List[EmailAttachment]
    headers: Dict[str, str]
    
    @property
    def full_text(self) -> str:
        """Get full email as searchable text."""
        parts = [
            f"From: {self.from_address}",
            f"To: {', '.join(self.to_addresses)}",
            f"Subject: {self.subject}",
            f"Date: {self.date.isoformat() if self.date else 'Unknown'}",
            "",
            self.body_text
        ]
        return "\n".join(parts)
    
    @property
    def has_attachments(self) -> bool:
        return len(self.attachments) > 0


class EmailParser:
    """
    Parse email messages from various formats.
    
    Supports:
    - EML files (single email)
    - MBOX files (multiple emails)
    - IMAP connection (live mailbox)
    """
    
    def __init__(
        self,
        extract_attachments: bool = True,
        max_attachment_size: int = 10 * 1024 * 1024  # 10MB
    ):
        self.extract_attachments = extract_attachments
        self.max_attachment_size = max_attachment_size
    
    def _decode_header_value(self, value: str) -> str:
        """Decode email header value (handles encoded words)."""
        if value is None:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(value):
            if isinstance(part, bytes):
                decoded_parts.append(
                    part.decode(encoding or 'utf-8', errors='replace')
                )
            else:
                decoded_parts.append(part)
        
        return ' '.join(decoded_parts)
    
    def _extract_addresses(self, header_value: str) -> List[str]:
        """Extract email addresses from header."""
        if not header_value:
            return []
        
        decoded = self._decode_header_value(header_value)
        # Simple regex for email addresses
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', decoded)
        return list(set(emails))
    
    def _get_body_text(self, msg: email.message.Message) -> str:
        """Extract plain text body from email."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        return payload.decode(charset, errors='replace')
            
            # Fallback: try to get HTML and convert
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        html = payload.decode(charset, errors='replace')
                        return self._html_to_text(html)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                text = payload.decode(charset, errors='replace')
                
                if msg.get_content_type() == 'text/html':
                    return self._html_to_text(text)
                return text
        
        return ""
    
    def _get_body_html(self, msg: email.message.Message) -> Optional[str]:
        """Extract HTML body from email."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        return payload.decode(charset, errors='replace')
        elif msg.get_content_type() == 'text/html':
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                return payload.decode(charset, errors='replace')
        
        return None
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        # Remove script and style
        html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
        html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)
        
        # Replace block elements with newlines
        html = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)
        
        # Remove tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        
        # Clean whitespace
        html = re.sub(r' +', ' ', html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        
        return html.strip()
    
    def _extract_attachments(self, msg: email.message.Message) -> List[EmailAttachment]:
        """Extract attachments from email."""
        attachments = []
        
        if not msg.is_multipart():
            return attachments
        
        for part in msg.walk():
            content_disposition = str(part.get('Content-Disposition', ''))
            
            if 'attachment' in content_disposition:
                filename = part.get_filename()
                if filename:
                    filename = self._decode_header_value(filename)
                else:
                    filename = 'unnamed_attachment'
                
                content_type = part.get_content_type()
                payload = part.get_payload(decode=True)
                size = len(payload) if payload else 0
                
                attachment = EmailAttachment(
                    filename=filename,
                    content_type=content_type,
                    size=size,
                    content=payload if self.extract_attachments and size <= self.max_attachment_size else None
                )
                attachments.append(attachment)
        
        return attachments
    
    def parse_message(self, msg: email.message.Message) -> ParsedEmail:
        """Parse an email.message.Message object."""
        # Parse date
        date_str = msg.get('Date')
        date = None
        if date_str:
            try:
                date = parsedate_to_datetime(date_str)
            except Exception:
                pass
        
        # Extract headers
        headers = {}
        for key in ['From', 'To', 'Cc', 'Subject', 'Date', 'Reply-To', 'Message-ID']:
            value = msg.get(key)
            if value:
                headers[key] = self._decode_header_value(value)
        
        return ParsedEmail(
            message_id=msg.get('Message-ID', ''),
            subject=self._decode_header_value(msg.get('Subject', '')),
            from_address=self._extract_addresses(msg.get('From', ''))[0] if msg.get('From') else '',
            to_addresses=self._extract_addresses(msg.get('To', '')),
            cc_addresses=self._extract_addresses(msg.get('Cc', '')),
            date=date,
            body_text=self._get_body_text(msg),
            body_html=self._get_body_html(msg),
            attachments=self._extract_attachments(msg),
            headers=headers,
        )
    
    def parse_eml(self, file_path: str) -> ParsedEmail:
        """Parse a single .eml file."""
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)
        return self.parse_message(msg)
    
    def parse_eml_bytes(self, content: bytes) -> ParsedEmail:
        """Parse email from bytes."""
        msg = email.message_from_bytes(content)
        return self.parse_message(msg)
    
    def parse_mbox(self, file_path: str) -> List[ParsedEmail]:
        """Parse an MBOX file (multiple emails)."""
        import mailbox
        
        mbox = mailbox.mbox(file_path)
        emails = []
        
        for msg in mbox:
            try:
                parsed = self.parse_message(msg)
                emails.append(parsed)
            except Exception as e:
                # Skip malformed messages
                continue
        
        return emails
    
    def fetch_imap(
        self,
        host: str,
        username: str,
        password: str,
        folder: str = "INBOX",
        limit: int = 100,
        ssl: bool = True
    ) -> List[ParsedEmail]:
        """
        Fetch emails from IMAP server.
        
        Args:
            host: IMAP server hostname
            username: Email username
            password: Email password
            folder: Folder to fetch from
            limit: Maximum number of emails to fetch
            ssl: Use SSL connection
        
        Returns:
            List of parsed emails
        """
        import imaplib
        
        if ssl:
            imap = imaplib.IMAP4_SSL(host)
        else:
            imap = imaplib.IMAP4(host)
        
        try:
            imap.login(username, password)
            imap.select(folder)
            
            # Search for all emails
            _, message_numbers = imap.search(None, 'ALL')
            message_ids = message_numbers[0].split()
            
            # Get latest 'limit' emails
            message_ids = message_ids[-limit:]
            
            emails = []
            for msg_id in message_ids:
                _, msg_data = imap.fetch(msg_id, '(RFC822)')
                email_body = msg_data[0][1]
                
                msg = email.message_from_bytes(email_body)
                parsed = self.parse_message(msg)
                emails.append(parsed)
            
            return emails
        finally:
            imap.logout()


def parse_email_file(file_path: str) -> str:
    """
    Convenience function to extract text from email file.
    
    Args:
        file_path: Path to .eml or .mbox file
    
    Returns:
        Extracted text as string
    """
    parser = EmailParser(extract_attachments=False)
    path = Path(file_path)
    
    if path.suffix.lower() == '.mbox':
        emails = parser.parse_mbox(file_path)
        return "\n\n---\n\n".join(e.full_text for e in emails)
    else:
        email_obj = parser.parse_eml(file_path)
        return email_obj.full_text

