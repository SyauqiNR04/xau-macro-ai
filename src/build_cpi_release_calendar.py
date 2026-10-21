
import csv
import re
import time
from pathlib import Path
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "processed" / "cpi_release_calendar.csv"

ARCHIVE_URL = "https://www.bls.gov/bls/news-release/cpi.htm"
START_YEAR = 2010

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

MONTHS = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


class ArchiveParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.parts = []
        self.page_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs = dict(attrs)
            self.href = attrs.get("href")
            self.parts = []

    def handle_data(self, data):
        self.page_text.append(data)
        if self.href is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.href is not None:
            self.links.append({
                "text": " ".join("".join(self.parts).split()),
                "url": self.href,
            })
            self.href = None
            self.parts = []


def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/155.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def fetch_archive_links(session):
    print("Mengakses indeks arsip CPI BLS...")

    response = session.get(ARCHIVE_URL, timeout=60)
    response.raise_for_status()

    parser = ArchiveParser()
    parser.feed(response.text)

    releases = {}

    for link in parser.links:
        match = re.search(
            r"^(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{4})"
            r"\s+Consumer Price Index$",
            link["text"],
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        month_name = match.group(1).capitalize()
        year = int(match.group(2))
        month = MONTHS[month_name]

        if year < START_YEAR:
            continue

        href_path = urlparse(link["url"]).path

        if not re.search(
            r"/cpi_\d{8}\.htm$",
            href_path,
            flags=re.IGNORECASE,
        ):
            continue

        date_match = re.search(
            r"cpi_(\d{2})(\d{2})(\d{4})\.htm$",
            href_path,
            flags=re.IGNORECASE,
        )

        if not date_match:
            continue

        mm, dd, yyyy = map(int, date_match.groups())

        try:
            release_date = datetime(yyyy, mm, dd).date()
        except ValueError:
            continue

        reference_month = f"{year}-{month:02d}-01"

        releases[reference_month] = {
            "reference_month": reference_month,
            "release_date": release_date.isoformat(),
            "source_url": urljoin(ARCHIVE_URL, link["url"]),
        }

    if not releases:
        raise RuntimeError(
            "Arsip CPI tidak ditemukan. Periksa koneksi atau format halaman BLS."
        )

    return releases


def extract_release_time(page_html):
    parser = ArchiveParser()
    parser.feed(page_html)

    text = unescape(" ".join(parser.page_text))
    text = re.sub(r"\s+", " ", text)

    # Cari waktu yang secara eksplisit tercantum pada header embargo.
    pattern = re.compile(
        r"embargoed\s+until\s+"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*"
        r"(?P<ampm>a\.m\.|p\.m\.|am|pm)\s*"
        r"\((?P<zone>ET|EDT|EST)\)",
        flags=re.IGNORECASE,
    )

    match = pattern.search(text)

    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    ampm = match.group("ampm").lower().replace(".", "")
    zone_label = match.group("zone").upper()

    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        return None

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    return {
        "hour": hour,
        "minute": minute,
        "zone_label": zone_label,
    }


def verify_release(session, release):
    release_date = datetime.strptime(
        release["release_date"], "%Y-%m-%d"
    ).date()

    result = {
        "reference_month": release["reference_month"],
        "release_date": release["release_date"],
        "release_time_et": "",
        "release_datetime_et": "",
        "release_datetime_utc": "",
        "status": "released_archive_found",
        "source_url": release["source_url"],
        "time_basis": "release_time_not_verified",
        "release_date_verified": True,
        "release_time_verified": False,
    }

    try:
        response = session.get(release["source_url"], timeout=30)
        response.raise_for_status()

        extracted = extract_release_time(response.text)

        if extracted is None:
            result["time_basis"] = "time_not_found_in_archive_header"
            return result

        # ET mengikuti zona waktu New York, termasuk perubahan DST.
        local_dt = datetime(
            release_date.year,
            release_date.month,
            release_date.day,
            extracted["hour"],
            extracted["minute"],
            tzinfo=ET,
        )

        # Jika arsip menyebut EST/EDT secara eksplisit, cocokkan dengan
        # offset zona waktu yang berlaku pada tanggal rilis tersebut.
        actual_abbr = local_dt.tzname()

        if (
            extracted["zone_label"] in {"EST", "EDT"}
            and extracted["zone_label"] != actual_abbr
        ):
            result["time_basis"] = (
                f"timezone_mismatch_header_{extracted['zone_label']}"
                f"_calculated_{actual_abbr}"
            )
            return result

        utc_dt = local_dt.astimezone(UTC)

        result.update({
            "release_time_et": local_dt.strftime("%H:%M:%S"),
            "release_datetime_et": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "release_datetime_utc": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_basis": f"verified_from_archive_header_{extracted['zone_label']}",
            "release_time_verified": True,
        })

        return result

    except requests.RequestException as exc:
        result["time_basis"] = (
            f"fetch_error_{type(exc).__name__}"
        )
        return result


def build_calendar(releases, session):
    latest_reference = max(releases)
    latest = datetime.strptime(latest_reference, "%Y-%m-%d")

    rows = []
    year, month = START_YEAR, 1

    total = len(releases)
    processed = 0

    while (year, month) <= (latest.year, latest.month):
        reference_month = f"{year}-{month:02d}-01"

        if reference_month in releases:
            processed += 1
            print(f"[{processed}/{total}] Memeriksa {reference_month}...")

            row = verify_release(session, releases[reference_month])

            # Beri jeda kecil agar tidak membebani server BLS.
            time.sleep(0.15)

        elif reference_month == "2025-10-01":
            row = {
                "reference_month": reference_month,
                "release_date": "",
                "release_time_et": "",
                "release_datetime_et": "",
                "release_datetime_utc": "",
                "status": "not_published",
                "source_url": ARCHIVE_URL,
                "time_basis": "not_published",
                "release_date_verified": False,
                "release_time_verified": False,
            }

        else:
            row = {
                "reference_month": reference_month,
                "release_date": "",
                "release_time_et": "",
                "release_datetime_et": "",
                "release_datetime_utc": "",
                "status": "unverified",
                "source_url": ARCHIVE_URL,
                "time_basis": "release_not_found_in_archive_index",
                "release_date_verified": False,
                "release_time_verified": False,
            }

        rows.append(row)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return rows


def main():
    session = make_session()
    releases = fetch_archive_links(session)
    rows = build_calendar(releases, session)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "reference_month",
        "release_date",
        "release_time_et",
        "release_datetime_et",
        "release_datetime_utc",
        "status",
        "source_url",
        "time_basis",
        "release_date_verified",
        "release_time_verified",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    verified = sum(row["release_time_verified"] for row in rows)
    known_dates = sum(row["release_date_verified"] for row in rows)

    print("\n=== HASIL VERIFIKASI KALENDER CPI ===")
    print(f"Total bulan: {len(rows)}")
    print(f"Tanggal rilis tersedia: {known_dates}")
    print(f"Waktu rilis terverifikasi: {verified}")
    print(f"Waktu belum terverifikasi: {len(rows) - verified}")
    print(f"File tersimpan: {OUTPUT_FILE}")

    print("\nContoh hasil terbaru:")
    for row in rows[-5:]:
        print(
            row["reference_month"],
            "| tanggal:", row["release_date"] or "-",
            "| waktu ET:", row["release_time_et"] or "-",
            "| waktu verified:", row["release_time_verified"],
            "| basis:", row["time_basis"],
        )


if __name__ == "__main__":
    main()
