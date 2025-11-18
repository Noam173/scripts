#!/usr/bin/env python3
"""
Improved speedtest-cli - Command line interface for testing internet bandwidth
"""

import csv
import errno
import json
import math
import os
import platform
import re
import signal
import socket
import ssl
import sys
import threading
import time
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from datetime import UTC, datetime
from http.client import BadStatusLine, HTTPConnection, HTTPSConnection
from io import BytesIO, StringIO
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import (
    AbstractHTTPHandler,
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPRedirectHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
    urlopen,
)

try:
    import gzip

    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False

__version__ = "3.0.0"

# Configuration constants
DEFAULT_TIMEOUT = 10
DEBUG = False


# ============================================================================
# Custom Exceptions
# ============================================================================


class SpeedtestException(Exception):
    """Base exception for this module"""


class SpeedtestCLIError(SpeedtestException):
    """Generic exception for raising errors during CLI operation"""


class SpeedtestHTTPError(SpeedtestException):
    """Base HTTP exception for this module"""


class SpeedtestConfigError(SpeedtestException):
    """Configuration XML is invalid"""


class ConfigRetrievalError(SpeedtestHTTPError):
    """Could not retrieve config.php"""


class ServersRetrievalError(SpeedtestHTTPError):
    """Could not retrieve speedtest-servers.php"""


class NoMatchedServers(SpeedtestException):
    """No servers matched when filtering"""


class SpeedtestBestServerFailure(SpeedtestException):
    """Unable to determine best server"""


class SpeedtestUploadTimeout(SpeedtestException):
    """Upload timeout reached"""


# ============================================================================
# HTTP Connection Classes
# ============================================================================


class SpeedtestHTTPConnection(HTTPConnection):
    """Custom HTTPConnection with source address support"""

    def __init__(
        self,
        *args,
        source_address: Optional[Tuple[str, int]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        self.source_address = source_address
        self.timeout = timeout
        super().__init__(*args, timeout=timeout, **kwargs)

    def connect(self):
        """Connect to the host with optional source address binding"""
        self.sock = socket.create_connection(
            (self.host, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()


class SpeedtestHTTPSConnection(HTTPSConnection):
    """Custom HTTPSConnection with source address support"""

    def __init__(
        self,
        *args,
        source_address: Optional[Tuple[str, int]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        context: Optional[ssl.SSLContext] = None,
        **kwargs,
    ):
        self.source_address = source_address
        self.timeout = timeout
        super().__init__(*args, timeout=timeout, context=context, **kwargs)

    def connect(self):
        """Connect to the host with optional source address binding"""
        self.sock = socket.create_connection(
            (self.host, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()

        if hasattr(self, "_context"):
            kwargs = {}
            if self._tunnel_host:
                kwargs["server_hostname"] = self._tunnel_host
            else:
                kwargs["server_hostname"] = self.host
            self.sock = self._context.wrap_socket(self.sock, **kwargs)


class SpeedtestHTTPHandler(AbstractHTTPHandler):
    """Custom HTTP handler with timeout and source address support"""

    def __init__(
        self,
        source_address: Optional[Tuple[str, int]] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__()
        self.source_address = source_address
        self.timeout = timeout

    def http_open(self, req):
        def build_conn(host, **kwargs):
            return SpeedtestHTTPConnection(
                host, source_address=self.source_address, timeout=self.timeout
            )

        return self.do_open(build_conn, req)


class SpeedtestHTTPSHandler(AbstractHTTPHandler):
    """Custom HTTPS handler with timeout and source address support"""

    def __init__(
        self,
        source_address: Optional[Tuple[str, int]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        context: Optional[ssl.SSLContext] = None,
    ):
        super().__init__()
        self.source_address = source_address
        self.timeout = timeout
        self._context = context or ssl.create_default_context()

    def https_open(self, req):
        def build_conn(host, **kwargs):
            return SpeedtestHTTPSConnection(
                host,
                source_address=self.source_address,
                timeout=self.timeout,
                context=self._context,
            )

        return self.do_open(build_conn, req)


# ============================================================================
# Utility Functions
# ============================================================================


def build_user_agent() -> str:
    """Build a Mozilla/5.0 compatible User-Agent string"""
    ua_parts = [
        "Mozilla/5.0",
        f"({platform.platform()}; {platform.architecture()[0]})",
        f"Python/{platform.python_version()}",
        "(KHTML, like Gecko)",
        f"speedtest-cli/{__version__}",
    ]
    return " ".join(ua_parts)


def build_opener(
    source_address: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT
) -> OpenerDirector:
    """Build URL opener with custom handlers"""
    source_tuple = (source_address, 0) if source_address else None

    handlers = [
        ProxyHandler(),
        SpeedtestHTTPHandler(source_address=source_tuple, timeout=timeout),
        SpeedtestHTTPSHandler(source_address=source_tuple, timeout=timeout),
        HTTPDefaultErrorHandler(),
        HTTPRedirectHandler(),
        HTTPErrorProcessor(),
    ]

    opener = OpenerDirector()
    opener.addheaders = [("User-Agent", build_user_agent())]

    for handler in handlers:
        opener.add_handler(handler)

    return opener


def build_request(
    url: str,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    bump: str = "0",
    secure: bool = False,
) -> Request:
    """Build a urllib request with cache busting"""
    headers = headers or {}

    # Add scheme if missing
    if url.startswith(":"):
        scheme = "https" if secure else "http"
        url = f"{scheme}{url}"

    # Add cache busting parameter
    delim = "&" if "?" in url else "?"
    final_url = f"{url}{delim}x={int(time.time() * 1000)}.{bump}"

    headers.update({"Cache-Control": "no-cache"})

    return Request(final_url, data=data, headers=headers)


def distance(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
    """Calculate distance between two lat/lon points in km using Haversine formula"""
    lat1, lon1 = origin
    lat2, lon2 = destination

    radius = 6371  # Earth's radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def print_debug(message: str):
    """Print debug message if DEBUG is enabled"""
    if DEBUG:
        print(
            f"\033[1;30mDEBUG: {message}\033[0m"
            if sys.stdout.isatty()
            else f"DEBUG: {message}",
            file=sys.stderr,
        )


# ============================================================================
# Threading Classes
# ============================================================================


class HTTPDownloader(threading.Thread):
    """Thread class for downloading data"""

    def __init__(
        self,
        index: int,
        request: Request,
        start_time: float,
        timeout: int,
        opener: OpenerDirector,
        shutdown_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.index = index
        self.request = request
        self.start_time = start_time
        self.timeout = timeout
        self.opener = opener
        self.shutdown_event = shutdown_event
        self.result = [0]

    def run(self):
        """Download data until timeout or completion"""
        try:
            if (time.time() - self.start_time) <= self.timeout:
                response = self.opener.open(self.request)

                while (
                    not self.shutdown_event.is_set()
                    and (time.time() - self.start_time) <= self.timeout
                ):
                    chunk = response.read(10240)
                    if not chunk:
                        break
                    self.result.append(len(chunk))

                response.close()
        except (IOError, HTTPError, URLError):
            pass


class HTTPUploaderData:
    """File-like object for upload data with timeout support"""

    def __init__(
        self,
        length: int,
        start_time: float,
        timeout: int,
        shutdown_event: threading.Event,
    ):
        self.length = length
        self.start_time = start_time
        self.timeout = timeout
        self.shutdown_event = shutdown_event
        self._data: Optional[BytesIO] = None
        self.total = [0]

    def pre_allocate(self):
        """Pre-allocate upload data"""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        multiplier = int(round(self.length / 36.0))
        content = f"content1={(chars * multiplier)[: self.length - 9]}"
        self._data = BytesIO(content.encode())

    @property
    def data(self) -> BytesIO:
        """Get data buffer, pre-allocating if needed"""
        if self._data is None:
            self.pre_allocate()
        return self._data

    def read(self, n: int = 10240) -> bytes:
        """Read data with timeout check"""
        if (
            time.time() - self.start_time
        ) <= self.timeout and not self.shutdown_event.is_set():
            chunk = self.data.read(n)
            self.total.append(len(chunk))
            return chunk
        raise SpeedtestUploadTimeout()


class HTTPUploader(threading.Thread):
    """Thread class for uploading data"""

    def __init__(
        self,
        index: int,
        request: Request,
        start_time: float,
        size: int,
        timeout: int,
        opener: OpenerDirector,
        shutdown_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.index = index
        self.request = request
        self.request.data.start_time = start_time
        self.start_time = start_time
        self.size = size
        self.timeout = timeout
        self.opener = opener
        self.shutdown_event = shutdown_event
        self.result = 0

    def run(self):
        """Upload data until timeout or completion"""
        try:
            if (
                time.time() - self.start_time
            ) <= self.timeout and not self.shutdown_event.is_set():
                response = self.opener.open(self.request)
                response.read(11)
                response.close()
                self.result = sum(self.request.data.total)
        except (IOError, SpeedtestUploadTimeout, HTTPError, URLError):
            self.result = sum(self.request.data.total)


# ============================================================================
# Results Class
# ============================================================================


class SpeedtestResults:
    """Container for speedtest results"""

    def __init__(
        self,
        download: float = 0,
        upload: float = 0,
        ping: float = 0,
        server: Optional[Dict] = None,
        client: Optional[Dict] = None,
    ):
        self.download = download
        self.upload = upload
        self.ping = ping
        self.server = server or {}
        self.client = client or {}
        self.timestamp = datetime.now(UTC).isoformat()
        self.bytes_received = 0
        self.bytes_sent = 0

    def __repr__(self) -> str:
        return repr(self.dict())

    def dict(self) -> Dict[str, Any]:
        """Return results as dictionary"""
        return {
            "download": self.download,
            "upload": self.upload,
            "ping": self.ping,
            "server": self.server,
            "timestamp": self.timestamp,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "client": self.client,
        }

    @staticmethod
    def csv_header(delimiter: str = ",") -> str:
        """Return CSV header"""
        headers = [
            "Server ID",
            "Sponsor",
            "Server Name",
            "Timestamp",
            "Distance",
            "Ping",
            "Download",
            "Upload",
            "IP Address",
        ]
        return delimiter.join(headers)

    def csv(self, delimiter: str = ",") -> str:
        """Return results as CSV"""
        data = self.dict()
        row = [
            str(data["server"].get("id", "")),
            data["server"].get("sponsor", ""),
            data["server"].get("name", ""),
            data["timestamp"],
            str(data["server"].get("d", "")),
            str(data["ping"]),
            str(data["download"]),
            str(data["upload"]),
            self.client.get("ip", ""),
        ]
        return delimiter.join(row)

    def json(self, pretty: bool = False) -> str:
        """Return results as JSON"""
        kwargs = {"indent": 4, "sort_keys": True} if pretty else {}
        return json.dumps(self.dict(), **kwargs)


# ============================================================================
# Main Speedtest Class
# ============================================================================


class Speedtest:
    """Main speedtest class"""

    def __init__(
        self,
        source_address: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        secure: bool = False,
        shutdown_event: Optional[threading.Event] = None,
    ):
        self.config: Dict[str, Any] = {}
        self._source_address = source_address
        self._timeout = timeout
        self._opener = build_opener(source_address, timeout)
        self._secure = secure
        self._shutdown_event = shutdown_event or threading.Event()

        self.servers: Dict[float, List[Dict]] = {}
        self.closest: List[Dict] = []
        self._best: Dict[str, Any] = {}

        self.get_config()
        self.results = SpeedtestResults(client=self.config.get("client", {}))

    @property
    def best(self) -> Dict[str, Any]:
        """Get best server, finding it if necessary"""
        if not self._best:
            self.get_best_server()
        return self._best

    def get_config(self):
        """Download speedtest.net configuration"""
        headers = {}
        if GZIP_AVAILABLE:
            headers["Accept-Encoding"] = "gzip"

        request = build_request(
            "://www.speedtest.net/speedtest-config.php",
            headers=headers,
            secure=self._secure,
        )

        try:
            response = self._opener.open(request)

            # Handle gzip encoding
            if response.headers.get("content-encoding") == "gzip":
                buf = BytesIO(response.read())
                with gzip.GzipFile(fileobj=buf) as f:
                    config_xml = f.read()
            else:
                config_xml = response.read()

            response.close()

            print_debug(f"Config XML:\n{config_xml.decode()}")

            root = ET.fromstring(config_xml)

            server_config = root.find("server-config").attrib
            download = root.find("download").attrib
            upload = root.find("upload").attrib
            client = root.find("client").attrib

            # Parse configuration
            ignore_servers = [
                int(i) for i in server_config["ignoreids"].split(",") if i
            ]

            ratio = int(upload["ratio"])
            upload_max = int(upload["maxchunkcount"])
            up_sizes = [32768, 65536, 131072, 262144, 524288, 1048576, 7340032]

            sizes = {
                "upload": up_sizes[ratio - 1 :],
                "download": [350, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
            }

            size_count = len(sizes["upload"])
            upload_count = int(math.ceil(upload_max / size_count))

            counts = {
                "upload": upload_count,
                "download": int(download["threadsperurl"]),
            }

            threads = {
                "upload": int(upload["threads"]),
                "download": int(server_config["threadcount"]) * 2,
            }

            length = {
                "upload": int(upload["testlength"]),
                "download": int(download["testlength"]),
            }

            self.config.update(
                {
                    "client": client,
                    "ignore_servers": ignore_servers,
                    "sizes": sizes,
                    "counts": counts,
                    "threads": threads,
                    "length": length,
                    "upload_max": upload_count * size_count,
                }
            )

            self.lat_lon = (float(client["lat"]), float(client["lon"]))

            print_debug(f"Config:\n{self.config}")

        except (HTTPError, URLError, ET.ParseError) as e:
            raise ConfigRetrievalError(f"Failed to retrieve configuration: {e}")

    def get_servers(
        self, servers: Optional[List[int]] = None, exclude: Optional[List[int]] = None
    ):
        """Retrieve list of speedtest.net servers"""
        servers = servers or []
        exclude = exclude or []
        self.servers.clear()

        urls = [
            "://www.speedtest.net/speedtest-servers-static.php",
            "://www.speedtest.net/speedtest-servers.php",
        ]

        headers = {}
        if GZIP_AVAILABLE:
            headers["Accept-Encoding"] = "gzip"

        for url in urls:
            try:
                request = build_request(
                    f"{url}?threads={self.config['threads']['download']}",
                    headers=headers,
                    secure=self._secure,
                )

                response = self._opener.open(request)

                # Handle gzip encoding
                if response.headers.get("content-encoding") == "gzip":
                    buf = BytesIO(response.read())
                    with gzip.GzipFile(fileobj=buf) as f:
                        servers_xml = f.read()
                else:
                    servers_xml = response.read()

                response.close()

                print_debug(f"Servers XML:\n{servers_xml.decode()}")

                root = ET.fromstring(servers_xml)

                for server in root.iter("server"):
                    attrib = server.attrib
                    server_id = int(attrib.get("id"))

                    # Filter servers
                    if servers and server_id not in servers:
                        continue
                    if (
                        server_id in self.config["ignore_servers"]
                        or server_id in exclude
                    ):
                        continue

                    try:
                        d = distance(
                            self.lat_lon, (float(attrib["lat"]), float(attrib["lon"]))
                        )
                        attrib["d"] = d

                        if d not in self.servers:
                            self.servers[d] = []
                        self.servers[d].append(attrib)
                    except (KeyError, ValueError):
                        continue

                break

            except (HTTPError, URLError, ET.ParseError):
                continue

        if (servers or exclude) and not self.servers:
            raise NoMatchedServers("No servers matched the filter criteria")

        return self.servers

    def get_closest_servers(self, limit: int = 5) -> List[Dict]:
        """Get closest servers by distance"""
        if not self.servers:
            self.get_servers()

        for d in sorted(self.servers.keys()):
            for server in self.servers[d]:
                self.closest.append(server)
                if len(self.closest) >= limit:
                    return self.closest

        return self.closest

    def get_best_server(self, servers: Optional[List[Dict]] = None) -> Dict:
        """Find server with lowest latency"""
        if not servers:
            servers = self.closest if self.closest else self.get_closest_servers()

        results = {}

        for server in servers:
            cum = []
            url = os.path.dirname(server["url"])
            stamp = int(time.time() * 1000)

            for i in range(3):
                latency_url = f"{url}/latency.txt?x={stamp}.{i}"

                try:
                    start = time.time()
                    response = self._opener.open(latency_url)
                    text = response.read(9)
                    total = time.time() - start
                    response.close()

                    if text == b"test=test":
                        cum.append(total)
                    else:
                        cum.append(3600)
                except (HTTPError, URLError, socket.error):
                    cum.append(3600)

            avg = round((sum(cum) / 6) * 1000, 3)
            results[avg] = server

        try:
            fastest = sorted(results.keys())[0]
        except IndexError:
            raise SpeedtestBestServerFailure(
                "Unable to connect to servers to test latency"
            )

        best = results[fastest]
        best["latency"] = fastest

        self.results.ping = fastest
        self.results.server = best
        self._best.update(best)

        print_debug(f"Best Server:\n{best}")

        return best

    def download(
        self,
        callback: Callable = lambda *args, **kwargs: None,
        threads: Optional[int] = None,
    ) -> float:
        """Test download speed"""
        urls = []
        for size in self.config["sizes"]["download"]:
            for _ in range(self.config["counts"]["download"]):
                urls.append(
                    f"{os.path.dirname(self.best['url'])}/random{size}x{size}.jpg"
                )

        request_count = len(urls)
        requests = [
            build_request(url, bump=str(i), secure=self._secure)
            for i, url in enumerate(urls)
        ]

        max_threads = threads or self.config["threads"]["download"]
        finished = []
        active_threads = []

        start = time.time()

        # Create and start threads
        for i, request in enumerate(requests):
            thread = HTTPDownloader(
                i,
                request,
                start,
                self.config["length"]["download"],
                self._opener,
                self._shutdown_event,
            )

            # Wait if we've hit max threads
            while len([t for t in active_threads if t.is_alive()]) >= max_threads:
                time.sleep(0.001)

            thread.start()
            active_threads.append(thread)
            callback(i, request_count, start=True)

        # Wait for all threads to complete
        for thread in active_threads:
            thread.join()
            finished.append(sum(thread.result))
            callback(thread.index, request_count, end=True)

        stop = time.time()

        self.results.bytes_received = sum(finished)
        self.results.download = (self.results.bytes_received / (stop - start)) * 8.0

        # Adjust upload threads based on download speed
        if self.results.download > 100000:
            self.config["threads"]["upload"] = 8

        return self.results.download

    def upload(
        self,
        callback: Callable = lambda *args, **kwargs: None,
        pre_allocate: bool = True,
        threads: Optional[int] = None,
    ) -> float:
        """Test upload speed"""
        sizes = []
        for size in self.config["sizes"]["upload"]:
            for _ in range(self.config["counts"]["upload"]):
                sizes.append(size)

        request_count = self.config["upload_max"]
        requests = []

        for i, size in enumerate(sizes[:request_count]):
            data = HTTPUploaderData(
                size, 0, self.config["length"]["upload"], self._shutdown_event
            )
            if pre_allocate:
                data.pre_allocate()

            headers = {"Content-Length": str(size)}
            requests.append(
                (
                    build_request(
                        self.best["url"],
                        data=data,
                        headers=headers,
                        secure=self._secure,
                    ),
                    size,
                )
            )

        max_threads = threads or self.config["threads"]["upload"]
        finished = []
        active_threads = []

        start = time.time()

        # Create and start threads
        for i, (request, size) in enumerate(requests):
            thread = HTTPUploader(
                i,
                request,
                start,
                size,
                self.config["length"]["upload"],
                self._opener,
                self._shutdown_event,
            )

            # Wait if we've hit max threads
            while len([t for t in active_threads if t.is_alive()]) >= max_threads:
                time.sleep(0.001)

            thread.start()
            active_threads.append(thread)
            callback(i, request_count, start=True)

        # Wait for all threads to complete
        for thread in active_threads:
            thread.join()
            finished.append(thread.result)
            callback(thread.index, request_count, end=True)

        stop = time.time()

        self.results.bytes_sent = sum(finished)
        self.results.upload = (self.results.bytes_sent / (stop - start)) * 8.0

        return self.results.upload


# ============================================================================
# CLI Functions
# ============================================================================


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(
        description="Command line interface for testing internet bandwidth"
    )

    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not perform download test",
    )
    parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Do not perform upload test",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use single connection instead of multiple",
    )
    parser.add_argument(
        "--bytes", action="store_true", help="Display values in bytes instead of bits"
    )
    parser.add_argument("--simple", action="store_true", help="Suppress verbose output")
    parser.add_argument("--csv", action="store_true", help="Output in CSV format")
    parser.add_argument(
        "--csv-delimiter", default=",", help="CSV delimiter (default: ,)"
    )
    parser.add_argument("--csv-header", action="store_true", help="Print CSV headers")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument(
        "--list", action="store_true", help="List servers sorted by distance"
    )
    parser.add_argument(
        "--server",
        type=int,
        action="append",
        help="Specify server ID (can be used multiple times)",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        action="append",
        help="Exclude server ID (can be used multiple times)",
    )
    parser.add_argument("--source", help="Source IP address to bind to")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--secure", action="store_true", help="Use HTTPS instead of HTTP"
    )
    parser.add_argument(
        "--no-pre-allocate",
        dest="pre_allocate",
        action="store_false",
        help="Do not pre-allocate upload data",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser.parse_args()


def print_dots(shutdown_event: threading.Event) -> Callable:
    """Create callback for printing progress dots"""

    def callback(current: int, total: int, start: bool = False, end: bool = False):
        if shutdown_event.is_set():
            return
        sys.stdout.write(".")
        if current + 1 == total and end:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return callback


def main():
    """Main CLI function"""
    global DEBUG

    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        shutdown_event.set()
        print("\nCancelling...", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    if args.version:
        print(f"speedtest-cli {__version__}")
        print(f"Python {platform.python_version()}")
        sys.exit(0)

    if args.csv_header:
        print(SpeedtestResults.csv_header(args.csv_delimiter))
        sys.exit(0)

    if not args.download and not args.upload:
        print("ERROR: Cannot disable both download and upload tests", file=sys.stderr)
        sys.exit(1)

    if len(args.csv_delimiter) != 1:
        print("ERROR: CSV delimiter must be a single character", file=sys.stderr)
        sys.exit(1)

    DEBUG = args.debug
    quiet = args.simple or args.csv or args.json

    callback = (lambda *a, **k: None) if quiet or DEBUG else print_dots(shutdown_event)

    # Units for display
    unit_name = "byte" if args.bytes else "bit"
    unit_divisor = 8 if args.bytes else 1

    if not quiet:
        print("Retrieving speedtest.net configuration...")

    try:
        speedtest = Speedtest(
            source_address=args.source,
            timeout=args.timeout,
            secure=args.secure,
            shutdown_event=shutdown_event,
        )
    except (ConfigRetrievalError, HTTPError, URLError) as e:
        print(f"ERROR: Cannot retrieve speedtest configuration: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        try:
            speedtest.get_servers()
        except (ServersRetrievalError, HTTPError, URLError) as e:
            print(f"ERROR: Cannot retrieve server list: {e}", file=sys.stderr)
            sys.exit(1)

        for _, servers in sorted(speedtest.servers.items()):
            for server in servers:
                print(
                    f"{server['id']:>5}) {server['sponsor']} "
                    f"({server['name']}, {server['country']}) "
                    f"[{server['d']:.2f} km]"
                )
        sys.exit(0)

    if not quiet:
        client = speedtest.config["client"]
        print(f"Testing from {client['isp']} ({client['ip']})...")
        print("Retrieving speedtest.net server list...")

    try:
        speedtest.get_servers(servers=args.server, exclude=args.exclude)
    except NoMatchedServers:
        print(
            f"ERROR: No matched servers: {', '.join(str(s) for s in args.server or [])}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ServersRetrievalError, HTTPError, URLError) as e:
        print(f"ERROR: Cannot retrieve server list: {e}", file=sys.stderr)
        sys.exit(1)

    if not quiet:
        if args.server and len(args.server) == 1:
            print("Retrieving information for selected server...")
        else:
            print("Selecting best server based on ping...")

    try:
        speedtest.get_best_server()
    except SpeedtestBestServerFailure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    results = speedtest.results

    if not quiet:
        print(
            f"Hosted by {results.server['sponsor']} ({results.server['name']}) "
            f"[{results.server['d']:.2f} km]: {results.ping} ms"
        )

    # Download test
    if args.download:
        if not quiet:
            print("Testing download speed", end="" if not DEBUG else "\n")

        speedtest.download(callback=callback, threads=1 if args.single else None)

        if not quiet:
            download_speed = (results.download / 1000 / 1000) / unit_divisor
            print(f"Download: {download_speed:.2f} M{unit_name}/s")
    else:
        if not quiet:
            print("Skipping download test")

    # Upload test
    if args.upload:
        if not quiet:
            print("Testing upload speed", end="" if not DEBUG else "\n")

        speedtest.upload(
            callback=callback,
            pre_allocate=args.pre_allocate,
            threads=1 if args.single else None,
        )

        if not quiet:
            upload_speed = (results.upload / 1000 / 1000) / unit_divisor
            print(f"Upload: {upload_speed:.2f} M{unit_name}/s")
    else:
        if not quiet:
            print("Skipping upload test")

    print_debug(f"Results:\n{results.dict()}")

    # Output results
    if args.simple:
        download_speed = (results.download / 1000 / 1000) / unit_divisor
        upload_speed = (results.upload / 1000 / 1000) / unit_divisor
        print(f"Ping: {results.ping} ms")
        print(f"Download: {download_speed:.2f} M{unit_name}/s")
        print(f"Upload: {upload_speed:.2f} M{unit_name}/s")
    elif args.csv:
        print(results.csv(delimiter=args.csv_delimiter))
    elif args.json:
        print(results.json())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelling...", file=sys.stderr)
        sys.exit(0)
    except SpeedtestException as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
