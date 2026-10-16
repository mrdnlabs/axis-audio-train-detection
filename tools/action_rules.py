"""Minimal client for the VAPIX Action Rules SOAP API (dev/test use).

  python tools/action_rules.py templates            # list action templates
  python tools/action_rules.py list                 # list action rules and configurations
"""
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import device as d  # noqa: E402

NS = ('xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:aa="http://www.axis.com/vapix/ws/action1" '
      'xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2" xmlns:tns1="http://www.onvif.org/ver10/topics" '
      'xmlns:tnsaxis="http://www.axis.com/2009/event/topics"')


def call(body):
    env = f'<?xml version="1.0" encoding="UTF-8"?><s:Envelope {NS}><s:Body>{body}</s:Body></s:Envelope>'
    r = d.S.post(d.BASE + "/vapix/services", data=env.encode(), timeout=30,
                 headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
    return r.text


def pretty(xml, limit=6000):
    return re.sub(r">\s*<", ">\n<", xml[xml.find("Body>"):])[:limit]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "templates":
        print(pretty(call("<aa:GetActionTemplates/>"), 20000))
    else:
        print(pretty(call("<aa:GetActionRules/>")))
        print(pretty(call("<aa:GetActionConfigurations/>")))
