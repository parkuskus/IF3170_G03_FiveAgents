# Booking Cancellation Prediction

## Overview

A canceled booking leaves an empty room that could have been sold to another guest. Predicting cancellations early helps an accommodation property plan overbooking, send reminders, and set suitable cancellation policies.

This competition provides booking records from an accommodation property. Each record describes the guests, length of stay, meal package, room category, booking channel, price, and booking history.

The goal is to build a classification model from scratch that predicts whether a booking is canceled (`is_canceled = 1`) or not canceled (`is_canceled = 0`).

## Workflow

1. Train the model on `train.csv`, which contains bookings and their outcomes.
2. Predict `is_canceled` for every booking in `test.csv`.
3. Submit predictions in the format of `sample_submission.csv`.

## Evaluation

Submissions are scored using the **Macro F1-score** between predicted labels and true labels. It is the average of the F1-scores for canceled and non-canceled bookings. Both classes count equally, so a model that predicts only one class receives a low score.

The public leaderboard uses about 30% of the test data. The final ranking uses the private leaderboard, which covers the remaining 70% and is revealed after the competition ends.

## Dataset

The dataset contains booking records from an accommodation property. Each row describes one booking, including the guests, stay, booking channel, and price. The target is whether the booking is canceled.

### Files

| File | Description |
| --- | --- |
| `train.csv` | Training set, including the `is_canceled` target. |
| `test.csv` | Test set, without the target. |
| `sample_submission.csv` | Example submission in the required format. |

### Columns

The `is_canceled` column appears only in `train.csv`.

| Column | Type | Description |
| --- | --- | --- |
| `id` | string | Unique booking identifier. Read as text, not as a number. |
| `num_adults` | integer | Number of adults. |
| `num_children` | integer | Number of children. |
| `weekend_nights` | integer | Number of weekend nights (Saturday or Sunday) booked. |
| `weekday_nights` | integer | Number of weekday nights (Monday to Friday) booked. |
| `meal_package` | categorical | Meal package: Room Only, Breakfast, Half Board (breakfast and one other meal), or Full Board (three meals a day). |
| `needs_parking` | binary | `1` if the guest needs a car parking space; otherwise `0`. |
| `room_category` | categorical | Room category, from Type A to Type G. The letters are anonymized and do not indicate order or quality. |
| `days_before_arrival` | integer | Number of days between the booking date and arrival date. |
| `arrival_year` | integer | Year of arrival. |
| `arrival_month` | integer | Month of arrival, from 1 to 12. |
| `arrival_day` | integer | Day of the month of arrival, from 1 to 31. |
| `booking_channel` | categorical | Booking channel: Online Travel Agent, Offline Agent, Corporate, Airline, or Complimentary. |
| `is_returning_guest` | binary | `1` if the guest has stayed at the property before; otherwise `0`. |
| `prior_cancellations` | integer | Number of the guest's previous bookings that were canceled. |
| `prior_completed_bookings` | integer | Number of the guest's previous bookings that were not canceled. |
| `nightly_rate_idr` | integer | Average price per room per night in IDR. Prices can differ for the same room category; `0` usually means a complimentary stay. |
| `special_requests` | integer | Number of special requests, such as a high floor or a particular view. |
| `is_canceled` | binary | Target: `1` if the booking is canceled; otherwise `0`. |

## Submission

Submit one row for every `id` in `test.csv`. The CSV must have a header and exactly two columns, `id` and `is_canceled`; the target must be the integer `0` or `1`.

```csv
id,is_canceled
5cb6f473,0
c06b4ab3,1
22bf3fc3,0
```

Some IDs contain only digits or may look like scientific notation (for example, `123e4567`). Read the `id` column as a string to preserve its value:

```python
pd.read_csv("test.csv", dtype={"id": str})
```

## Notes

- IDR stands for Indonesian Rupiah.
- Some feature values are missing and appear as empty cells in `train.csv` and `test.csv`.

## Citation

rzi. *Tugas Besar 2 IF3170*. [Kaggle competition](https://www.kaggle.com/competitions/T2-IF3170-2026-2027), 2026.
