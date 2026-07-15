# Flash-Flood Web-Verified Sources 2026-07-14

This file documents a first pass of externally sourced flash-flood events collected from publicly accessible web pages on July 14, 2026.

Input file:

- `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv`

Scope:

- The rows are intended to exercise the repository's verified-ingestion contract with real public citations.
- This is not yet a production-grade official source baseline.
- Some rows rely on stable event summary pages when the original article URL was unavailable or no longer fetchable from the current browsing environment.
- Jeddah duplicate rows no longer need exact seed-coordinate alignment to merge; normalized location and date range now drive duplicate folding unless a geometry is supplied.

Source quality notes:

- `2009-11-25 Jeddah`: sourced from the `2009 Jeddah floods` event page, which cites BBC, AFP, Saudi Gazette, and Arab News.
- `2011-01-26 Jeddah`: sourced from the `Jeddah` floods section, which cites CNN and Arab News for the 2011 flood episode.
- `2015-11-17 Jeddah`: citation URL from Al Arabiya listed in the `Jeddah` floods references.
- `2017-11-21 Jeddah`: citation URL from Saudi Gazette listed in the `Jeddah` floods references.
- `2022-11-24 Jeddah`: directly fetched Straits Times article confirming two deaths, school closures, and flight delays in Jeddah.
- `2022-12-23 Mecca`: sourced from the `2022-2023 Saudi Arabia floods` event page, which records a Mecca flash-flood episode on December 23, 2022.
- `2023-01-03 Jeddah`: sourced from the same `2022-2023 Saudi Arabia floods` event page, which records renewed flooding in Jeddah in early January 2023.

Recommended interpretation:

- Treat this CSV as `web_verified_candidate` quality rather than a final official bulletin baseline.
- Keep provenance fields intact during ingestion.
- Prefer upgrading each row later with a direct official or primary-media citation when available.
