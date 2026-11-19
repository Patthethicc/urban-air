# other-country-scrapers

Hourly PM2.5 of stations in Vietnam, Thailand, Indonesia and Malaysia from the **OpenAQ API v3**, written in the same layout as the Manila data, for the transfer-learning experiments
(`model/transfer/`, `transfer_learning.ipynb`, `xgboost/training/transfer.py`). PM2.5 history only: no weather, no morphology.

| File | What it is |
|---|---|
| `scrapers.ipynb` | the notebook (VS Code / Jupyter): settings, find stations, estimate the cost, download (resumable), keep the good stations, plot what was kept |
| `openaq_scraper.py` | the code behind it: client with rate limiting and retries, station discovery, download, quality rules, writer |
| `tests/test_openaq_scraper.py` | tests against a fake API (no key, no network): `python -m pytest other-country-scrapers/tests -q` |

## Sites scraped

Only **OpenAQ** (`https://api.openaq.org/v3`): `/locations` (the PM2.5 stations inside a city's bounding box) and `/sensors/{id}/hours` (hourly values). Cities: Ho Chi Minh City and Hanoi (Vietnam),
Bangkok (Thailand), Jakarta (Indonesia), Kuala Lumpur (Malaysia); add more in `openaq_scraper.CITIES`. Each is the capital region of its country (the counterpart of Manila's NCR): Hanoi, Bangkok and its metropolitan ring, Jakarta and its Greater Jakarta neighbours, Kuala Lumpur and the Klang Valley. Ho Chi Minh City (not a capital, but the largest city of Vietnam) is fetched as well, so both the capital regions and the big cities are covered. The Zenodo "HCMC / Bangkok hourly" datasets are **not** used: they are CAMS model output, not measurements.

## Use

1. Get a free key at <https://explore.openaq.org/register> and set it in the environment (never in a file): PowerShell `$env:OPENAQ_API_KEY = "..."`.
2. Open `scrapers.ipynb`, run the cells with `DRY_RUN = True` (lists stations and estimates the requests: free keys allow 60 a minute and 2,000 an hour), then set `DRY_RUN = False`.
3. Output: `data/Other Countries/<country>/<city>/` with `sensors/openaq_<id>_<name>.csv` (columns `location_name, location_id, sensor_id, utc_timestamp, local_timestamp, pm25_value, parameter, unit`),
   `locations.csv`, `station_quality.csv` (every station and why it was kept or dropped) and `source.json` (UTM zone, time zone, range, rules). Raw downloads are cached in `_raw/` so a stopped run resumes.

A station is kept with at least 6,000 hours, readings in 12 different calendar months and 50 % of the hours between its first and last reading present (all adjustable in the notebook).
Hours are labelled by their start (`TIME_LABEL = "from"`): compare one value with the OpenAQ portal and with a Manila file before mixing the networks. Check each provider's licence before redistributing the data.

## If OpenAQ answers "408 Connection timed out"

The server gives up when one query asks for too much. The scraper therefore fetches each sensor in **60-day windows** (`window_days`), saves every window on its own under `_raw/<city>/<sensor>.parts/`, and **halves a window that still times out** (down to one day). A stopped or failed run resumes at the window it was on, not at the start of the sensor, and a station is only asked for the dates between its own first and last reading. Just run the download cell again. If a single day still times out, the error names the sensor; skip it or try again later (the OpenAQ API is sometimes simply slow).

## Extending the date range later

Change `END` in the notebook (it is exclusive: `"2026-11-19"` includes all of 18 November) and run the download cell again. A sensor that was already downloaded is not fetched from the start: only the days after its last hour are requested and appended, and a station whose own last reading is already covered costs no request.
