# MLOps Zenith — M25CSA032

Data contracts for ML and analytics pipelines. This repository holds YAML data contracts defined with the [Data Contract Specification](https://datacontract.com/) (version 0.9.3) to ensure schema stability, validation, and clear ownership between producers and consumers.

## Overview

Each contract describes:

- **Schema** — field names, types, constraints, and PII flags  
- **Quality rules** — expectations and validation (e.g. patterns, ranges, enums)  
- **Metadata** — owner, contacts, classification, and tags  

Use these contracts to validate data at ingestion, document APIs/datasets, and avoid breaking changes for downstream models and dashboards.

## Contracts

| Contract | Description | Domain |
|----------|-------------|--------|
| **fintech_contract.yaml** | Bank Transaction Log — real-time transaction records for fraud detection. Enforces strict account ID format (10-char `A-Z0-9`). | Fintech, Banking |
| **orders_contract.yaml** | Black Friday Flash Sale Orders — real-time order stream for marketing dashboards. Enforces non-negative `order_total_usd` and mapped status enum (`PAID`, `SHIPPED`, `CANCELLED`). | E-commerce |
| **rides_contract.yaml** | CityMove Ride-Share Rides — operational ride data for dynamic pricing ML. Defines stable field names (e.g. `fare_final`), freshness SLA, and PII handling for `passenger_id`. | Ride-share, ML |
| **thermostat_contract.yaml** | Smart Thermostat Fleet Telemetry — temperature and battery telemetry from a fleet of smart thermostats for reporting and predictive maintenance. Validates temperature range (-30°C–60°C) and device identifiers. | IoT |

## Contract structure (Data Contract Spec 0.9.3)

- **info** — `title`, `description`, `owner`, `contact`, `classification`, `tags`
- **schema** — JSON Schema-style `properties` with types, patterns, enums, min/max, and `pii` flags
- **quality** (where present) — expectations and rules for data validation

## Usage

- **Producers:** Emit or store data that conforms to these schemas and run validation (e.g. schema + quality checks) before publishing.
- **Consumers:** Use the contracts as the single source of truth for field names, types, and constraints when building pipelines, models, or dashboards.
- **Tooling:** Validate payloads or files against the YAML contracts using a Data Contract–compatible validator or custom scripts that load the YAML and apply the schema/quality rules.

## Repository

- **Course/Project:** MLOps Zenith  
- **Identifier:** M25CSA032  

