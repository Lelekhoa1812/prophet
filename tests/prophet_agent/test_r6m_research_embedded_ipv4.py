"""r6m embedded IPv4: an IPv6 address that carries a private IPv4 address is refused as that IPv4 address.

Defect: Python reports the NAT64 well-known prefix (64:ff9b::/96) and IPv4-compatible addresses (::/96) as global, so a
name whose AAAA record is 64:ff9b::7f00:1 (127.0.0.1) or 64:ff9b::a9fe:a9fe (169.254.169.254) passed the address check.
On a NAT64 network the gateway would connect to the embedded IPv4 address, which can be a private or metadata address.
No network: the resolver is injected.
"""
import pytest

from prophet_agent.research.netguard import UnsafeURL, blocked_reason, check_url


@pytest.mark.parametrize("address, reason", [
    ("64:ff9b::7f00:1", "loopback"),            # 127.0.0.1
    ("64:ff9b::a9fe:a9fe", "metadata"),         # 169.254.169.254
    ("64:ff9b::a00:1", "private"),              # 10.0.0.1
    ("64:ff9b::c0a8:101", "private"),           # 192.168.1.1
    ("::7f00:1", "loopback"),                   # IPv4-compatible 127.0.0.1
    ("::a9fe:a9fe", "metadata"),                # IPv4-compatible 169.254.169.254
    ("64:ff9b:1::5db8:d822", "non-public"),     # RFC 8215 local-use NAT64 prefix is never fetched
])
def test_ipv6_addresses_that_embed_a_private_ipv4_are_refused(address, reason):
    assert blocked_reason(address) == reason


def test_nat64_address_of_a_public_ipv4_is_allowed():
    assert blocked_reason("64:ff9b::5db8:d822") is None      # 93.184.216.34


def test_hostname_whose_answer_embeds_the_metadata_address_is_refused():
    with pytest.raises(UnsafeURL, match="metadata"):
        check_url("https://looks-public.example.com/", resolver=lambda host, port: ["64:ff9b::a9fe:a9fe"])


def test_plain_ipv6_public_address_is_still_allowed():
    assert blocked_reason("2606:4700:4700::1111") is None
