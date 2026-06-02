"""UPnP ConnectionManager SOAP handling."""
from __future__ import annotations

SOURCE_PROTOCOL_INFO = (
    "http-get:*:video/mp4:*,"
    "http-get:*:video/MP2T:*,"
    "http-get:*:video/x-matroska:*"
)

SERVICE_TYPE = "urn:schemas-upnp-org:service:ConnectionManager:1"


def _wrap_soap(action: str, body_xml: str) -> bytes:
    env = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action}Response xmlns:u="{SERVICE_TYPE}">'
        f"{body_xml}"
        f"</u:{action}Response>"
        "</s:Body></s:Envelope>"
    )
    return env.encode("utf-8")


def _fault() -> bytes:
    env = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        "<s:Body><s:Fault><faultcode>s:Client</faultcode>"
        "<faultstring>UPnPError</faultstring><detail>"
        '<UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
        "<errorCode>401</errorCode><errorDescription>Invalid Action</errorDescription>"
        "</UPnPError></detail></s:Fault></s:Body></s:Envelope>"
    )
    return env.encode("utf-8")


def handle_soap(soap_action: str, body: bytes) -> tuple[bytes, int]:
    """Return a SOAP response for ConnectionManager actions."""
    del body
    action = soap_action.strip('"').split("#")[-1]

    if action == "GetProtocolInfo":
        return _wrap_soap(
            "GetProtocolInfo",
            f"<Source>{SOURCE_PROTOCOL_INFO}</Source><Sink></Sink>",
        ), 200

    if action == "GetCurrentConnectionIDs":
        return _wrap_soap(
            "GetCurrentConnectionIDs",
            "<ConnectionIDs>0</ConnectionIDs>",
        ), 200

    if action == "GetCurrentConnectionInfo":
        return _wrap_soap(
            "GetCurrentConnectionInfo",
            "<RcsID>-1</RcsID>"
            "<AVTransportID>-1</AVTransportID>"
            "<ProtocolInfo></ProtocolInfo>"
            "<PeerConnectionManager></PeerConnectionManager>"
            "<PeerConnectionID>-1</PeerConnectionID>"
            "<Direction>Output</Direction>"
            "<Status>Unknown</Status>",
        ), 200

    return _fault(), 401
