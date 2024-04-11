import numpy as np
import pandas as pd


def get_transition_distribution(num_users, max_seq_len, train, time_thresh):
    # Compute the transition distribution mentioned in  4.2
    # def compute_relation(train, num_users, max_seq_len, time_thresh):
    R = {}
    for u in range(num_users):
        # get the temporal components of user u of their last `max_seq_len` checkins
        time_seq = np.zeros([max_seq_len], dtype=np.int32)
        tmp = (train[u][-max_seq_len:][:, 1][-max_seq_len:]).astype(int)
        time_seq[-len(tmp):] = tmp
        time_seq = time_seq.reshape((len(time_seq), 1))

        # get the temporal difference of different POIs
        ones_vec = np.ones([1, len(time_seq)])
        mat1 = time_seq @ ones_vec
        mat2 = (time_seq @ ones_vec).T
        time_diff_matrix = np.absolute(mat1 - mat2)

        # cut difference that exceed threshold
        time_diff_matrix[time_diff_matrix > time_thresh] = time_thresh

        # save to a relation table
        R[u] = time_diff_matrix
    return R


def haversine_dist(lon1, lat1, lon2, lat2):
    # compute the haversine distance of two coordinates
    R = 6373.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    d = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    cons = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))
    return R * cons


def get_distance_distribution(num_users, max_seq_len, train, dis_thresh):
    # Compute the transition distribution mentioned in  4.2
    # def compute_relation(train, num_users, max_seq_len, time_thresh):
    D = {}
    for u in range(num_users):
        # get the distance components of user u of their last `max_seq_len` checkins
        seq = np.array(['0,0']*max_seq_len, dtype = 'object')
        tmp = train[u][-max_seq_len:][:, 2][-max_seq_len:]
        seq[-len(tmp):] = tmp[:]

        size = len(seq)
        seq = seq.reshape((len(seq), 1))

        df = pd.DataFrame({'lon':[float(x[0].split(',')[0]) for x in seq], 'lat':[float(x[0].split(',')[1]) for x in seq]})
        # cross join the longitude and latitude
        df['key'] = 0
        df = df.merge(df, on='key', how='outer')
        df['dist'] = haversine_dist(df['lon_x'], df['lat_x'], df['lon_y'], df['lat_y'])
        df['dist'] = df['dist'].abs().astype(int)
        df[df['dist'] > dis_thresh] = dis_thresh
        df[(df['lon_x'] == 0) & (df['lat_x'] == 0)] = dis_thresh
        df[(df['lon_y'] == 0) & (df['lat_y'] == 0)] = dis_thresh
        D[u] = df['dist'].values.reshape((size,size))
    return D