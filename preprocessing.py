import pandas as pd
import numpy as np

data = pd.read_csv('listings-full.csv', sep=',')
data['price'] = data['price'].str.replace('$','').str.replace(',','').astype(float)
data['security_deposit'] = data['security_deposit'].str.replace('$','').str.replace(',','').astype(float)
data['extra_people'] = data['extra_people'].str.replace('$','').str.replace(',','').astype(float)

df = data.select_dtypes(include=np.number)

selected = ['host_since', 'host_response_time', 'host_acceptance_rate', 'host_is_superhost', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'neighbourhood', 'neighbourhood_cleansed', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet', 'weekly_price', 'monthly_price', 'security_deposit', 'security_deposit', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'number_of_reviews_ltm', 'first_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'reviews_per_month', 'price', 'room_type']

cols = set(list(df.columns) + selected)

df = data[list(cols)]

df['num_amenities'] = df['amenities'].apply(lambda x: len(x.split(',')))

df.dropna(axis=1, how='all', inplace=True)

df = df[df.columns[df.count(axis=0)/len(df) > 0.5]]

to_drop = ['id', 'security_deposit', 'minimum_nights_avg_ntm', 'maximum_maximum_nights', 'neighbourhood_cleansed', 'host_id', 'scrape_id', 'host_since', 'host_response_time', 'maximum_minimum_nights', 'latitude',  'minimum_maximum_nights', 'minimum_minimum_nights', 'cancellation_policy', 'host_neighbourhood', 'maximum_nights_avg_ntm', 'amenities', 'longitude']

dff = df.drop(columns=to_drop)

dff = dff.replace(['f','t'],[0,1])

dff = dff[~dff['neighbourhood'].isna()]


dff.fillna(0, inplace=True)


#dff = pd.concat([dff, pd.get_dummies(dff['bed_type'], prefix='is')], axis=1)
#dff = pd.concat([dff, pd.get_dummies(dff['room_type'], prefix='is')], axis=1)

#dff = dff.drop(columns=['bed_type', 'room_type'])

dff.count(axis=0)/len(dff)

dff.dtypes

dff.to_csv('data2.csv', index=False)
