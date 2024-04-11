import pandas as pd
import numpy as np


def preprocess_and_load_dataset():
    # load the data
    df = pd.read_csv('datasets/foursquare_given/four-sin.txt',sep='\t', header=None)
    df.rename(columns={0:'user', 1:"poi", 2:'loc', 3:'timestamp'}, inplace=True)

    # This file preprocesses the datasets based on the following steps outlined in the paper:
    # 1. remove unpopular POIs with < 5 records
    # 2. filter inactive users with < 5 POI checkins
    while True:
        # include only users with at least 5 POIs
        user_counter = df.groupby('user')['user'].count()
        mapper_dict = {k: v for k, v in user_counter.items()}
        df['eligible'] = df['user'].apply(lambda x: mapper_dict[x])
        df['eligible'] = (df['eligible'] >= 5).values
        df = df[df['eligible'] == True]

        updated_num_entries = len(df)

        # include only POIs with at least 5 users
        count_locations = df.groupby('poi')['poi'].count()
        mapper_dict = {k: v for k, v in count_locations.items()}
        df['eligible'] = df['poi'].apply(lambda x: mapper_dict[x])
        df = df[df['eligible'] >= 5]

        if len(df) == updated_num_entries: break

    num_users = df['user'].unique().shape[0]
    num_pois = df['poi'].unique().shape[0]
    print('total number of users: ', num_users)
    print('total number of pois: ', num_pois)

    # renumber 'timestamp' and 'user' to start from 0
    df['timestamp'] = df['timestamp'].astype(str).astype(int)
    df['user'] = df['user'].astype(str).astype(int)
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['user'] = df['user'] - df['user'].min()

    num_times = max(df['timestamp'].unique())

    # Use leave-one-out strategy for dataset partition as mentioned in the paper.
    # Chronologically sort the check-in sequence of each user and hold out the last
    # two check-ins for validation and testing respectively, the remaining are utilized
    # as the training set.
    train_set, valid_set, test_set = {}, {}, {}
    user_groups = df.groupby('user')

    train, test, valid = {}, {}, {}
    for user, group in user_groups:
        t = df[df['user'] == user]['timestamp'].values
        p = df[df['user'] == user]['poi'].values
        l = df[df['user'] == user]['loc'].values

        vals = np.array([p, t, l]).T
        train[user] = vals[:-2]
        valid[user] = vals[-2]
        test[user] = vals[-1]

    return train, valid, test, num_users, num_pois, num_times
