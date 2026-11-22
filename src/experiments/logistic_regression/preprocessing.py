import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_CONFIG = {'month_category': True,
 'clip_extremes': True,
 'lead_flags': True,
 'online_interactions': True,
 'guest_flags': True,
 'drop_day': True,
 'price_flags': False,
 'category_interactions': False,
 'numeric_curves': False,
 'enhancements': ['splines',
                  'channel_bins',
                  'request_bins',
                  'counts',
                  'season']}

def build_base_features(raw, options):
    frame = raw.copy()
    if options['guest_flags']:
        frame['has_children'] = frame['num_children'].gt(0).astype(float).where(frame['num_children'].notna())
        frame['has_requests'] = frame['special_requests'].gt(0).astype(float).where(frame['special_requests'].notna())
        frame['has_weekend'] = frame['weekend_nights'].gt(0).astype(float)
        frame['total_nights'] = frame['weekday_nights'] + frame['weekend_nights']
    if options['lead_flags']:
        lead = frame['days_before_arrival']
        for boundary in [7, 30, 90, 180]:
            frame[f'lead_over_{boundary}'] = lead.gt(boundary).astype(float).where(lead.notna())
    if options['online_interactions']:
        online = frame['booking_channel'].eq('Online Travel Agent').astype(float)
        for column in ['days_before_arrival', 'nightly_rate_idr', 'special_requests']:
            frame[f'online_{column}'] = frame[column] * online
    if options['month_category']:
        frame['arrival_month'] = frame['arrival_month'].astype(str)
    if options['drop_day']:
        frame = frame.drop(columns=['arrival_day'])
    if options['price_flags']:
        rate = frame['nightly_rate_idr']
        for boundary in [500000, 1000000, 1500000, 2000000]:
            frame[f'rate_over_{boundary}'] = rate.gt(boundary).astype(float).where(rate.notna())
    if options['category_interactions']:
        channel = frame['booking_channel'].fillna('Missing').astype(str)
        for column in ['room_category', 'meal_package']:
            frame[f'channel_{column}'] = channel + ' | ' + frame[column].fillna('Missing').astype(str)
    if options['numeric_curves']:
        lead = frame['days_before_arrival']
        rate = frame['nightly_rate_idr']
        requests = frame['special_requests']
        nights = frame['weekend_nights'] + frame['weekday_nights']
        frame['lead_squared'] = lead ** 2
        frame['rate_squared'] = rate ** 2
        frame['lead_rate'] = lead * rate
        frame['lead_requests'] = lead * requests
        frame['rate_nights'] = rate * nights
        frame['requests_nights'] = requests * nights
    return frame

def build_features(raw, options):
    frame = build_base_features(raw, options)
    enabled = set(options.get('enhancements', []))
    lead = raw['days_before_arrival']
    rate = raw['nightly_rate_idr']
    requests = raw['special_requests']
    nights = raw['weekday_nights'] + raw['weekend_nights']
    guests = raw['num_adults'] + raw['num_children']
    channel = raw['booking_channel'].fillna('Missing').astype(str)
    if 'missing' in enabled:
        for col in ['num_adults', 'num_children', 'needs_parking', 'days_before_arrival',
                    'nightly_rate_idr', 'special_requests', 'meal_package', 'room_category']:
            frame[f'missing_{col}'] = raw[col].isna().astype(float)
    if 'history' in enabled:
        canceled = raw['prior_cancellations']
        completed = raw['prior_completed_bookings']
        frame['had_cancellation'] = canceled.gt(0).astype(float)
        frame['had_completed_booking'] = completed.gt(0).astype(float)
        frame['past_cancel_fraction'] = canceled / (canceled + completed).clip(lower=1)
        frame['log_completed'] = np.log1p(completed)
        frame['log_canceled'] = np.log1p(canceled)
    if 'stay' in enabled:
        frame['total_guests'] = guests
        frame['single_guest'] = guests.eq(1).astype(float).where(guests.notna())
        frame['rate_per_guest'] = rate / guests.clip(lower=1)
        frame['total_stay_price'] = rate * nights
        frame['weekend_fraction'] = raw['weekend_nights'] / nights.clip(lower=1)
        frame['requests_per_guest'] = requests / guests.clip(lower=1)
    if 'counts' in enabled:
        for col, cap in [('special_requests', 3), ('num_adults', 3), ('num_children', 2),
                         ('weekend_nights', 3), ('weekday_nights', 7)]:
            frame[f'category_{col}'] = raw[col].clip(upper=cap).fillna(-1).astype(str)
    # basis piecewise linear / hinge (James et al., ISLR, 2021)
    if 'splines' in enabled or 'channel_splines' in enabled:
        for col, values, knots in [
            ('lead', lead, [3, 7, 14, 30, 60, 90, 120, 180, 270]),
            ('rate', rate / 1000000, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]),
        ]:
            for knot in knots:
                hinge = (values - knot).clip(lower=0)
                if 'splines' in enabled:
                    frame[f'{col}_hinge_{knot}'] = hinge
                if 'channel_splines' in enabled:
                    for label, value in [('online', 'Online Travel Agent'), ('offline', 'Offline Agent')]:
                        frame[f'{label}_{col}_hinge_{knot}'] = hinge * channel.eq(value).astype(float)
    if enabled & {'bins', 'channel_bins', 'room_bins', 'request_bins', 'bin_grid'}:
        lead_bin = pd.cut(lead, [-np.inf, 1, 7, 14, 30, 60, 90, 120, 180, 270, np.inf]).astype(str).fillna('Missing')
        rate_bin = pd.cut(rate / 1000000, [-np.inf, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, np.inf]).astype(str).fillna('Missing')
        if 'bins' in enabled:
            frame['lead_bin'] = lead_bin
            frame['rate_bin'] = rate_bin
        if 'channel_bins' in enabled:
            frame['channel_lead_bin'] = channel + ' | ' + lead_bin
            frame['channel_rate_bin'] = channel + ' | ' + rate_bin
        if 'room_bins' in enabled:
            frame['room_rate_bin'] = raw['room_category'].fillna('Missing').astype(str) + ' | ' + rate_bin
        if 'request_bins' in enabled:
            request_bin = requests.clip(upper=3).fillna(-1).astype(str)
            frame['lead_request_bin'] = lead_bin + ' | ' + request_bin
            frame['channel_request_bin'] = channel + ' | ' + request_bin
        if 'bin_grid' in enabled:
            frame['lead_rate_bin'] = lead_bin + ' | ' + rate_bin
    if 'season' in enabled:
        month = raw['arrival_month'].astype(str)
        frame['channel_month'] = channel + ' | ' + month
        frame['year_month'] = raw['arrival_year'].astype(str) + ' | ' + month
    if 'date' in enabled:
        dates = pd.to_datetime(dict(year=raw['arrival_year'], month=raw['arrival_month'], day=raw['arrival_day']), errors='coerce')
        frame['arrival_weekday'] = dates.dt.dayofweek.fillna(-1).astype(str)
    return frame

def clip_features(frame, bounds):
    frame = frame.copy()
    if bounds:
        columns = list(bounds['lower'])
        frame[columns] = frame[columns].clip(lower=pd.Series(bounds['lower']), upper=pd.Series(bounds['upper']), axis=1)
    return frame

# fit hanya pada data latih: https://scikit-learn.org/stable/common_pitfalls.html
def fit_preprocessing(raw, options):
    frame = build_features(raw, options)
    binary = ['needs_parking', 'is_returning_guest']
    categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    numeric = [c for c in frame if c not in binary + categorical]
    bounds = None
    if options['clip_extremes']:
        quantile = options.get('clip_quantile', 0.005)
        bounds = {'lower': frame[numeric].quantile(quantile).to_dict(),
                  'upper': frame[numeric].quantile(1 - quantile).to_dict()}
    frame = clip_features(frame, bounds)
    transformer = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), numeric),
        ('binary', SimpleImputer(strategy='most_frequent', keep_empty_features=True), binary),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], remainder='drop')
    X = transformer.fit_transform(frame).astype(np.float64)
    assert np.isfinite(X).all()
    return transformer, bounds, X

def transform_preprocessing(raw, options, transformer, bounds):
    X = transformer.transform(clip_features(build_features(raw, options), bounds)).astype(np.float64)
    assert np.isfinite(X).all()
    return X
