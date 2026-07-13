# Layer-4 Label Sources

This document defines the current real-label strategy for Saudi Layer-4 risk modeling. It is intentionally hazard-specific. The project should not assume one shared label source or one shared supervision granularity across `extreme_heat`, `dry_heat_agriculture`, and `flash_flood`.

## Status

Current training code can fit hazard-specific LightGBM models, but the shipped targets are still rule-derived pseudo labels. Real supervision is only planned at this stage. The first hazard to migrate should be `flash_flood`.

## Hazard Routes

| Hazard | Recommended supervision unit | Priority | Current status |
| --- | --- | --- | --- |
| `flash_flood` | `grid-day` or `province-day` event label | Highest | Ready for event-table + mapping implementation |
| `extreme_heat` | `impact-region-day` focused on Hajj / Makkah | Medium | Needs impact table design and region scope definition |
| `dry_heat_agriculture` | `region-season` or `region-year` outcome label | Medium | Needs task redefinition before model training |

## Flash Flood

### Candidate sources

| Source family | Type | Expected coverage | Notes |
| --- | --- | --- | --- |
| EM-DAT | Disaster event database | National / subnational events | Good for dated flood events, but geometry is usually coarse |
| Geo-Disasters | Geocoded disaster events | Event point / area enrichment | Useful to add coordinates or region footprints to EM-DAT-like events |
| Saudi civil defense / official bulletins | Official event confirmation | Event-specific | Best source for manual validation when available |
| Peer-reviewed Saudi flood case studies | Secondary validation | Event-specific | Useful for event timing, location, and footprint refinement |

### Seed events already identified

These are handoff-approved seed events for the first table build:

| Date | Location | Event note |
| --- | --- | --- |
| 2009-11-25 | Jeddah | flash flood |
| 2011-01-26 | Jeddah | flash flood |
| 2015-11-17 | Jeddah | flash flood |
| 2017-11-21 | Jeddah | flash flood |
| 2022-11-24 | Jeddah | flash flood |
| 2022-12-23 | Mecca | flash flood |

### Required event-table schema

The first production label table should be a normalized CSV or Parquet file with these columns:

| Column | Meaning |
| --- | --- |
| `event_id` | Stable event identifier |
| `hazard_type` | Always `flash_flood` for this table |
| `start_date` | Event start date |
| `end_date` | Event end date |
| `location_name` | Human-readable place name |
| `country_code` | ISO code, expected `SAU` |
| `latitude` | Event center latitude when point-based |
| `longitude` | Event center longitude when point-based |
| `geometry_wkt` | Optional polygon / buffered geometry |
| `spatial_confidence` | `high`, `medium`, or `low` |
| `temporal_confidence` | `high`, `medium`, or `low` |
| `source_name` | Database or authority name |
| `source_url` | Traceable source link |
| `source_record_id` | Upstream record identifier |
| `validation_status` | `seed`, `verified`, or `rejected` |
| `notes` | Short deterministic note only |

### Mapping rules to training samples

The first mapping pass should stay simple and inspectable:

1. Expand each event to inclusive daily rows from `start_date` to `end_date`.
2. If a validated polygon exists, assign positive labels to grid cells intersecting the polygon.
3. If only a point exists, buffer the point with a configured radius and assign positives to intersecting cells.
4. If only region text exists, map to a province-day label instead of pretending to have grid precision.
5. Keep ambiguous rows as `label_status=uncertain` rather than forcing negative labels.

### Positive / negative / uncertain policy

| Label state | Definition |
| --- | --- |
| Positive | Cell or region intersects a validated event footprint during the event window |
| Negative | Cell or region is outside all validated event footprints for that day within the modeled domain |
| Uncertain | Event timing or geometry is too weak to support a confident spatial negative |

### Configuration requirements

These choices should live in config, not inside ad-hoc scripts:

- point-buffer radius
- province fallback enable flag
- uncertainty handling policy
- event duration expansion policy
- minimum source confidence for positive labels

## Extreme Heat

### Candidate sources

| Source family | Type | Expected coverage | Notes |
| --- | --- | --- | --- |
| Hajj heat-health incident reporting | Health impact event data | Makkah / Mina / Arafat / Muzdalifah | Strong real-world impact label but geographically concentrated |
| Ministry / official public-health statements | Official impact confirmation | Event-specific | Best for temporal validation |
| Peer-reviewed Hajj heat-health studies | Secondary evidence | Event-specific | Useful for counts and timing, not nationwide coverage |

### Recommended task

Do not frame this as nationwide daily grid classification yet. The first real-label task should be a Makkah-focused `impact-region-day` or site-cluster-day supervision problem centered on Hajj periods.

### Suggested label schema

- `date`
- `region_id`
- `impact_count`
- `impact_level`
- `population_context`
- `source_name`
- `source_url`
- `validation_status`

## Dry Heat Agriculture

### Candidate sources

| Source family | Type | Expected coverage | Notes |
| --- | --- | --- | --- |
| FAOSTAT | Agricultural outcome statistics | National / subnational depending series | Good for annual outcome supervision |
| Saudi official agriculture statistics | Official production statistics | National / regional | Preferred when spatial detail is available |
| Agricultural census / ministry reports | Structured seasonal or annual outcomes | Regional / crop-specific | Likely needed for crop and area detail |

### Recommended task

Do not continue assuming a nationwide daily event label. The likely viable supervision units are:

- `region-season`
- `region-year`
- crop-specific yield anomaly regression or classification

### Suggested label schema

- `year`
- `season`
- `region_id`
- `crop_type`
- `yield_value`
- `yield_anomaly`
- `harvest_area`
- `source_name`
- `source_url`
- `validation_status`

## Implementation Order

1. Build and validate the `flash_flood` event table.
2. Add an event-to-grid or event-to-province mapping script that emits a trainable label table.
3. Train the first real-label `flash_flood` model.
4. Define a dedicated `extreme_heat` impact table for Hajj-focused supervision.
5. Redefine `dry_heat_agriculture` as a seasonal or annual outcome task before collecting labels.

## Non-Goals

- No shared three-hazard label warehouse assumption.
- No forced `grid-day` framing for `dry_heat_agriculture`.
- No silent promotion of weak text-only evidence into precise grid labels.
