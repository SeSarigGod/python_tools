import asyncio
import re
import socket
import ssl
import email.message as emessage
import smtplib

# create SMTP container
_email = "sesariggod@gmail.com"
_pword = "xiju ntpa nfwx tble"
container = smtplib.SMTP_SSL(host='smtp.google.com', port=465)
container.login(user=_email, password=_pword)
print("Logged In")

CARRIER_MAP = {
    "verizon": "vtext.com",
    "tmobile": "tmomail.net",
    "sprint": "messaging.sprintpcs.com",
    "at&t": "txt.att.net",
    "boost": "smsmyboostmobile.com",
    "cricket": "sms.cricketwireless.net",
    "uscellular": "email.uscc.net",
}

def send_txt(num: str | int, carrier: str, email: str, pword: str, msg: str, subj: str) -> tuple[dict, str]:
    to_email = CARRIER_MAP[carrier]

    # build message
    message = emessage.EmailMessage()
    message["From"] = email
    message["To"] = f"{num}@{to_email}"
    message["Subject"] = subj
    message.set_content(msg)

    # send
    send_kws = dict(username=email, password=pword, start_tls=True)
    res = container.send_message(message, **send_kws)
    msg = "failed" if not re.search(r"\sOK\s", res[1]) else "succeeded"
    print(msg)
    return res


if __name__ == "__main__":
    _num = "3193336344"
    _carrier = "verizon"
    _msg = "Dummy msg"
    _subj = "Dummy subj"
    # send_txt(_num, _carrier, _email, _pword, _msg, _subj)