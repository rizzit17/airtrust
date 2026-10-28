# Prompt to send to a teammate

We are building a college project titled **Trust-Weighted Federated Learning for City-Wide Air Pollution Mapping in Delhi**. Please research suitable public datasets and return verified download links.

## What we are building

Multiple geographic sensor groups act as separate clients. Each client trains a pollution-prediction model locally. A central aggregator combines their model updates using federated learning, without receiving raw training records. We compare standard FedAvg with a proposed method that weights each client's update using calibration confidence: how well its sensors agree with trusted co-located reference monitors.

We predict PM2.5 concentration and produce a spatial prediction surface. Evaluation uses MAE, RMSE and R² on future times and stations excluded from fitting. Comparators include centralized neural training, kriging, IDW and robust aggregation. We also measure model-update communication and run a limited membership-inference probe. This is a software simulation on one laptop, not a physical sensor installation or a certified privacy system.

Real reference observations can support virtual low-cost sensors with disclosed bias, noise, drift and dropout. Actual paired low-cost/reference measurements would help us make those assumptions realistic. We must separately submit every dataset used, so downloadable files and clear redistribution rights matter.

## Existing data so you can avoid duplicating work

We have already downloaded calendar-year 2024 Delhi hourly PM2.5 observations and monitor coordinates from **India Air Quality Database, XKDR Forum**, based on CPCB observations. There are 317,627 source rows across 39 catalog entries; our current cleaning retains 36 distinct locations. The compilation is CC BY 4.0. Source: https://airquality.xkdr.org ; documentation: https://github.com/xKDR/Air-Quality-Database .

We also have the **UCI Beijing Multi-Site Air Quality** source archive, covering 12 reference stations during 2013–2017, as a secondary sanity benchmark. Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data .

Please prioritize complementary data rather than downloading those same sources again.

## Priorities

1. **Paired low-cost and reference PM2.5 readings**, preferably co-located in Delhi or another Indian city. Plantower/PMS, SDS011 or comparable sensor calibration datasets and published supplementary files would be useful. We need actual paired readings, not just claims that a sensor was calibrated.
2. **Real Delhi low-cost sensor histories**, especially from sensor.community or published studies. Confirm actual Delhi coverage, coordinates, timestamps, device types and downloadable history.
3. **Matching hourly Delhi meteorology for 2024**: relative humidity, temperature, wind speed/direction, pressure and rainfall. Verify local-time/UTC conventions, spatial resolution and license. Explain whether observations or modeled/reanalysis values are provided.
4. **Documented calibration/drift evidence** that can justify simulation assumptions. Find the original peer-reviewed paper, its actual parameter table and any downloadable data. Do not invent hardware parameters or attribute our chosen synthetic settings to a paper that does not report them.
5. If time permits, **independent Delhi station-year data or official boundary geometry** with a clear license, which could support stronger spatial evaluation and mask prediction grids to the city boundary.

## What to report for each candidate

- Title, original publisher, source page and a direct download link.
- Whether you actually tested the download, plus date checked.
- License and whether attributed redistribution with our college submission is permitted.
- Real date coverage, sampling interval, station count, approximate file size and format.
- Available columns, units, coordinates, device/reference type and missing-data issues.
- Whether access requires an account, API key, payment, CAPTCHA or manual approval.
- Exact role it could play in our project and any matching/joining problems.
- For paired sensors: distance between instruments, shared time coverage and how reference quality is established.

Return the best **3–5 verified options**, ranked by usefulness, with a recommendation. Distinguish confirmed facts from assumptions and inaccessible sources. Avoid city-average daily AQI-only data: our model needs station-level PM2.5 concentrations. Do not describe reference monitors as real low-cost sensors, or treat mirrored copies as independent datasets.
