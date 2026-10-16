"""Watch TrainPresent events through an ONVIF pull-point subscription.

  python tools/watch_events.py [seconds] [--inject path/to/16k_mono.wav]

Prints each event message (property operation, active, horn, test, confidence).
With --inject, plays the WAV into the live detector first (analyze.cgi?mode=inject).
"""
import re
import sys
import time
import uuid

sys.path.insert(0, __import__("os").path.dirname(__file__))
import device as d  # noqa: E402

NS = ('xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:wsa="http://www.w3.org/2005/08/addressing" '
      'xmlns:tev="http://www.onvif.org/ver10/events/wsdl" xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2" '
      'xmlns:tns1="http://www.onvif.org/ver10/topics" xmlns:tnsaxis="http://www.axis.com/2009/event/topics"')
TOPIC = "tnsaxis:CameraApplicationPlatform/TrainDetect/TrainPresent"


def soap(body, to=None, action=None, sub_id=None):
    hdr = ""
    path = "/vapix/services"
    if to:
        ref = (f'<dom0:SubscriptionId xmlns:dom0="http://www.axis.com/2009/event" wsa:IsReferenceParameter="true">'
               f'{sub_id}</dom0:SubscriptionId>') if sub_id else ""
        hdr = (f"<s:Header><wsa:Action>{action}</wsa:Action><wsa:MessageID>urn:uuid:{uuid.uuid4()}</wsa:MessageID>"
               f"<wsa:To>{to}</wsa:To>{ref}</s:Header>")
        # Same pull point through the VAPIX endpoint: it accepts the device login, while
        # /onvif/services requires a separate ONVIF user.
    env = f'<?xml version="1.0" encoding="UTF-8"?><s:Envelope {NS}>{hdr}<s:Body>{body}</s:Body></s:Envelope>'
    r = d.S.post(d.BASE + path, data=env.encode(), timeout=30,
                 headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    r.raise_for_status()
    return r.text


def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 90
    inject = sys.argv[sys.argv.index("--inject") + 1] if "--inject" in sys.argv else None
    resp = soap(f'<tev:CreatePullPointSubscription><tev:Filter><wsnt:TopicExpression '
                f'Dialect="http://www.onvif.org/ver10/tev/topicExpression/ConcreteSet">{TOPIC}</wsnt:TopicExpression>'
                f'</tev:Filter><tev:InitialTerminationTime>PT{secs + 60}S</tev:InitialTerminationTime>'
                f'</tev:CreatePullPointSubscription>')
    to = re.search(r"Address>([^<]+)</", resp).group(1)
    sid = re.search(r"SubscriptionId[^>]*>([^<]+)<", resp)
    sid = sid.group(1) if sid else None
    print("subscribed:", to)
    if inject:
        r = d.S.post(d.BASE + "/local/train_detect/analyze.cgi?mode=inject", data=open(inject, "rb").read(),
                     headers={"Content-Type": "audio/wav"}, timeout=60)
        print("inject:", r.status_code, r.text[:80])
    t0 = time.time()
    act = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription/PullMessagesRequest"
    while time.time() - t0 < secs:
        text = soap("<tev:PullMessages><tev:Timeout>PT5S</tev:Timeout><tev:MessageLimit>10</tev:MessageLimit>"
                    "</tev:PullMessages>", to, act, sid)
        for msg in re.findall(r"<wsnt:NotificationMessage>(.*?)</wsnt:NotificationMessage>", text, re.S):
            op = re.search(r'PropertyOperation="(\w+)"', msg)
            items = dict(re.findall(r'SimpleItem Name="(\w+)" Value="([^"]*)"', msg))
            utc = re.search(r'UtcTime="([^"]+)"', msg)
            print(f"+{time.time() - t0:5.1f}s {utc.group(1) if utc else ''} {op.group(1) if op else '?':12s} {items}",
                  flush=True)
    soap("<wsnt:Unsubscribe/>", to, "http://docs.oasis-open.org/wsn/bw-2/SubscriptionManager/UnsubscribeRequest", sid)


if __name__ == "__main__":
    main()
