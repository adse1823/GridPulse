# Risk Aggregation

## What this step does

Takes the outputs of the demand forecast and renewable forecast and answers:

> **For each hour in the next 48 hours, is supply likely to fall short of demand?**

This is deliberately **not ML**. It is plain arithmetic on top of two learned
forecasts. Keeping the decision logic simple and auditable is a design choice —
if the risk flag is wrong, the cause is traceable to either a bad demand
forecast, a bad renewable forecast, or a bad dispatchable capacity assumption.
A third model here would obscure that.

---

## The supply equation

```
total_supply(hour) = renewable_forecast(hour) + dispatchable_available(hour)
```

**Renewable forecast** comes from the wind + solar models.

**Dispatchable available** = the generation that can be scheduled and controlled
(gas peakers, combined cycle gas, nuclear, coal). For ERCOT Tier 1, we treat
this as a fixed number per season — a simplification that is wrong in detail
but close enough for a 48h ahead forecast:

| Season | Dispatchable available (approx) |
|--------|---------------------------------|
| Summer (Jun–Sep) | 46 GW |
| Winter (Nov–Feb) | 42 GW |
| Shoulder (Mar–May, Oct) | 50 GW |

These are conservative estimates based on EIA installed capacity minus typical
planned outage rates (~15%). The Feb 2021 event is the reason winter is lower —
cold weather trips gas plants that aren't weatherized.

**Why not forecast dispatchable capacity too?** Because it requires plant-level
outage data that isn't cleanly available in public APIs. In Tier 2 we can
incorporate ERCOT's own capacity adequacy reports. For Tier 1, fixed seasonal
estimates are honest and defensible.

---

## The risk flag

```
shortfall(hour) = demand_forecast(hour) - total_supply(hour)

if shortfall(hour) > SAFETY_MARGIN:
    flag hour as AT RISK
else:
    flag hour as OK
```

**Safety margin:** 3,000 MW (~3.5% of peak demand)

Grid operators use a "reserve margin" — they require supply to exceed demand
by a buffer to handle unexpected generator trips or demand forecast errors.
ERCOT's target reserve margin is ~13.75% in formal planning, but for a 48h
ahead operational forecast a 3 GW buffer is a reasonable rule of thumb.

This can be tuned — a larger margin catches more real risks but also produces
more false alarms. We use 3 GW as a starting point.

---

## Example output

```
Hour                  Demand   Wind    Solar   Dispatch  Supply   Shortfall  Flag
2024-01-15 17:00 CST  62,100   5,200     400    46,000   51,600    10,500   AT RISK
2024-01-15 18:00 CST  61,800   4,900     100    46,000   51,000    10,800   AT RISK
2024-01-15 19:00 CST  59,400   5,100       0    46,000   51,100     8,300   AT RISK
2024-01-15 22:00 CST  54,200   6,800       0    46,000   52,800     1,400   OK
2024-01-16 14:00 CST  47,800  10,200   3,100    46,000   59,300   -11,500   OK
```

Negative shortfall = surplus (supply > demand). Positive shortfall = deficit.

---

## Calibration check

After building the models, we can back-test the risk flags against known
historical events:

- **Feb 2021 Texas blackouts (Feb 10–19, 2021):** if we had run this pipeline
  on Feb 8 using forecasted weather, would it have flagged Feb 10–15 as AT RISK?
  This is the primary sanity check. The event is well documented — demand hit
  ~70 GW while dispatchable capacity dropped to ~45 GW due to equipment failures.

- **Aug 2023 heat dome:** ERCOT set a new all-time demand record (~85 GW).
  The pipeline should flag the peak hours.

We are not backtesting on these periods during training — we note them as
post-hoc validation cases after the full pipeline is built.

---

## Output format (Tier 1)

Plain text, no LLM:

```
=== GridPulse Risk Report ===
Generated: 2024-01-15 08:00 CST
Forecast window: 2024-01-15 09:00 → 2024-01-17 08:00

AT-RISK HOURS (17 of 48):
  2024-01-15 17:00–20:00 CST  |  Peak shortfall: 10,800 MW  |  Driver: high demand + low wind
  2024-01-16 07:00–12:00 CST  |  Peak shortfall:  4,200 MW  |  Driver: morning ramp + low solar

All other hours: sufficient margin.
```

The LLM narrative ("resembles the pattern behind the Feb 2021 event") is Tier 2.

---

## Files

```
agents/risk_aggregator/
└── aggregator.py   — takes demand_forecast df + renewable_forecast df,
                      returns per-hour risk table
agents/reporting/
└── report.py       — formats risk table as plain-text report (Tier 1)
```
