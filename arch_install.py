#! /usr/bin/env -S uv run

from datetime import datetime

import requests
from wget import download


def main() -> None:
    now = datetime.now()
    year = now.year
    month = now.month
    date = f"{year}.{month:02d}.01"
    base_url = "https://archlinux.interhost.co.il/iso/"
    url = f"{base_url}/{date}"
    response = requests.get(url)
    if response.status_code == 404:
        if month == 1:
            year = year - 1
            month = 12
        else:
            month = month - 1

        date = f"{year}.{month:02d}.01"
        url = f"{base_url}/{date}"
    answer = input(
        f"the latest arch iso version is {date}, do you wanna proceed? [Y/n] \n"
    )
    if answer == "n":
        print("Aborting")
    else:
        iso = f"{url}/archlinux-{date}-x86_64.iso"
        download(url=iso, out="arch.iso")
        print(f'\n\nDownloaded from {date=}')


if __name__ == "__main__":
    main()
