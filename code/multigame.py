import os
import boto3
import pandas as pd
import bz2
import json
import io
import math
import numpy as np
from tqdm import tqdm

# Manually configure AWS IAM credentials
os.environ["AWS_ACCESS_KEY_ID"] = "REDACTED"  
os.environ["AWS_SECRET_ACCESS_KEY"] = "REDACTED" 
os.environ["AWS_REGION"] = "us-east-1"  # Default aws region

S3_BUCKET = "shot-probability"
s3_client = boto3.client("s3")

def get_obj(file_path):
    return s3_client.get_object(Bucket=S3_BUCKET, Key=file_path)['Body']

import numpy as np

def tangent_inter(x0, y0, a, b, r, x_line):
    dx = x0 - a
    dy = y0 - b
    d = np.hypot(dx, dy)
    if d < r or np.isclose(d, r): # Inside or on the circle
        return [-10, 10]

    angle_to_point = np.arctan2(dy, dx)
    angle_offset = np.arcsin(r / d)
    angles = [angle_to_point + angle_offset, angle_to_point - angle_offset]

    points = []
    for theta in angles:
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        t = (x_line - x0) / dir_x
        y = y0 + t * dir_y
        points.append(y)        
    return sorted(points)


# competition, season, game = 'pl', '2022-2023', '4436'
def process_game(competition, season, game):
    print(f"Processing {competition} {season} {game}")
    file_path = f"pff-data/tracking/{competition}/{season}/{game}/{game}.jsonl.bz2"
    s3_body = get_obj(file_path)

    with bz2.BZ2File(s3_body) as bz2_file:
        with io.TextIOWrapper(bz2_file, encoding='utf-8') as f:
            try:
                data = [json.loads(line) for line in f]
            except EOFError:
                print(f"Corrupted bz2 file: {file_path}")
                return None

    df = pd.DataFrame(data)
    meta = pd.read_json(get_obj(f'pff-data/tracking/{competition}/{season}/{game}/metadata.json'), lines=True)
    rosters = pd.read_json(get_obj(f'pff-data/tracking/{competition}/{season}/{game}/rosters.json'))

    df_cleaned = df.dropna(subset=['homePlayersSmoothed', 'awayPlayersSmoothed']).reset_index(drop=True)
    home_count = df_cleaned.homePlayersSmoothed.apply(lambda x: 0 if x is None else len(x))
    away_count = df_cleaned.awayPlayersSmoothed.apply(lambda x: 0 if x is None else len(x))
    df_cleaned = df_cleaned[(home_count <= 11) & (away_count <= 11)].reset_index(drop=True)
    df_cleaned['in_play'] = True
    in_play = False
    for i in tqdm(range(len(df_cleaned))):
        if df_cleaned.game_event[i] is not None:
            event_type = df_cleaned.game_event[i]['game_event_type']
            if event_type in ['OUT', 'END']:
                in_play = False
            elif event_type in ['FIRSTKICKOFF', 'SECONDKICKOFF', 'OTB', 'G']:
                in_play = True
        df_cleaned.loc[i, 'in_play'] = in_play

    home_id = meta.homeTeam[0]['id']
    away_id = meta.awayTeam[0]['id']

    rosters['id'] = rosters.player.apply(lambda x: x['id'])
    rosters['name'] = rosters.player.apply(lambda x: x['nickname'])

    home_rosters = rosters[rosters['team'].apply(lambda x: x['id'] == home_id)].drop(columns=['player', 'team']).reset_index(drop=True)
    away_rosters = rosters[rosters['team'].apply(lambda x: x['id'] == away_id)].drop(columns=['player', 'team']).reset_index(drop=True)
    home_rosters['id'] = home_rosters['id'].astype(int)
    away_rosters['id'] = away_rosters['id'].astype(int)

    df_cleaned = df_cleaned[df_cleaned['in_play']].reset_index(drop=True)
    df_cleaned['is_home'] = df_cleaned['game_event'].apply(lambda x: None if x is None or x['game_event_type'] != 'OTB' else x['home_ball'])
    df_cleaned['period'] -= 1
    # 1: Attack from right to left, 0: Attack from left to right
    df_cleaned['attack_direction'] = None if df_cleaned.is_home is None else df_cleaned.is_home ^ meta.homeTeamStartLeft[0] ^ df_cleaned.period

    df_cleaned['x'] = df_cleaned['ballsSmoothed'].apply(lambda ball: None if ball is None or not isinstance(ball['x'], (int,float)) else ball['x'])
    df_cleaned['y'] = df_cleaned['ballsSmoothed'].apply(lambda ball: None if ball is None or not isinstance(ball['y'], (int,float)) else ball['y'])
    df_cleaned['z'] = df_cleaned['ballsSmoothed'].apply(lambda ball: None if ball is None or not isinstance(ball['z'], (int,float)) else ball['z'])

    df_cleaned['x'] = df_cleaned['x'].interpolate(method='linear')
    df_cleaned['y'] = df_cleaned['y'].interpolate(method='linear')
    df_cleaned['z'] = df_cleaned['z'].interpolate(method='linear')

    third_x = 52.5 - 105 / 3
    counter, is_home = 0, -1
    attack_start, attack_end = [], []
    df_cleaned['attack'] = 0

    def cleared(bx, atk_dir):
        return bx > -third_x if atk_dir else bx < third_x

    for i in tqdm(range(len(df_cleaned))):
        bx, by = df_cleaned.x[i], df_cleaned.y[i]
        if is_home == 1: # Home attack
            if not df_cleaned.is_home[i] or cleared(bx, df_cleaned.attack_direction[i]):
                is_home = -1
                attack_end.append(i)
        elif is_home == 0: # Away attack
            if df_cleaned.is_home[i] or cleared(bx, df_cleaned.attack_direction[i]):
                is_home = -1
                attack_end.append(i)
        else: # Not during an attack
            if (df_cleaned.is_home[i] is not None) and ((df_cleaned.attack_direction[i] and bx <= -third_x) or (not df_cleaned.attack_direction[i] and bx >= third_x)):
                is_home = df_cleaned.is_home[i]
                counter += 1
                attack_start.append(i)
        if is_home >= 0:
            df_cleaned.loc[i, 'attack'] = counter

    attacks = df_cleaned[df_cleaned.attack > 0].reset_index(drop=True)
    attacks['is_shot'] = attacks.possession_event.apply(lambda x: x['possession_event_type'] == 'SH' if x is not None else False)
    shots = attacks[attacks['is_shot']]
    attacks['hasShotsIn10s'] = False
    attacks['hasShotsIn5s'] = False
    attacks['hasShotsIn3s'] = False
    attacks['hasShotsIn1s'] = False
    for i in range(len(shots)):
        shotTime = shots.iloc[i, :].videoTimeMs
        attacks.loc[(attacks['videoTimeMs'] >= shotTime - 10000) & (attacks['videoTimeMs'] <= shotTime), 'hasShotsIn10s'] = True
        attacks.loc[(attacks['videoTimeMs'] >= shotTime - 5000) & (attacks['videoTimeMs'] <= shotTime), 'hasShotsIn5s'] = True
        attacks.loc[(attacks['videoTimeMs'] >= shotTime - 3000) & (attacks['videoTimeMs'] <= shotTime), 'hasShotsIn3s'] = True
        attacks.loc[(attacks['videoTimeMs'] >= shotTime - 1000) & (attacks['videoTimeMs'] <= shotTime), 'hasShotsIn1s'] = True
    
    attacks['x_flipped'] = attacks['x'] * (1 - 2 * attacks['attack_direction'])
    attacks['y_flipped'] = attacks['y'] * (1 - 2 * attacks['attack_direction'])
    attacks['r'] = attacks.apply(lambda ball: math.sqrt((52.5 - ball['x_flipped'])**2 + ball['y_flipped']**2), axis=1)
    attacks['theta'] = attacks.apply(lambda ball: math.atan2(ball['y_flipped'], (52.5 - ball['x_flipped'])), axis=1)

    attacks['time_s'] = attacks['videoTimeMs'] / 1000.0

    dx = attacks['x'].diff()
    dy = attacks['y'].diff()
    dt = attacks['time_s'].diff()
    attacks['speed'] = np.sqrt(dx**2 + dy**2) / dt
    attacks.loc[dt == 0, 'speed'] = np.nan
    attacks['speed'] = attacks['speed'].fillna(0)
    attacks['speed_smooth'] = attacks['speed'].rolling(window=5, center=True).mean()

    attacks['homePlayersRelative'] = [[] for _ in range(len(attacks))]
    attacks['awayPlayersRelative'] = [[] for _ in range(len(attacks))]
    attacks['GK_r'], attacks['GK_theta'], attacks['openGoal'] = 0.0, 0.0, 0.0

    for frame in tqdm(range(len(attacks))):
        cover_l, cover_r = [], []
        for i in range(len(attacks.homePlayersSmoothed[frame])):
            pl = attacks.homePlayersSmoothed[frame][i]
            x_flipped = pl['x'] * (1 - 2 * attacks.attack_direction[frame])
            y_flipped = pl['y'] * (1 - 2 * attacks.attack_direction[frame])
            r = math.sqrt((52.5 - x_flipped)**2 + y_flipped**2)
            theta = math.atan2(y_flipped, (52.5 - x_flipped))
            delta_x, delta_y = x_flipped-attacks.x_flipped[frame], y_flipped-attacks.y_flipped[frame]
            dist_ball = math.sqrt(delta_x**2 + delta_y**2)
            angle_ball = math.atan2(delta_x, delta_y)
            attacks.loc[frame, 'homePlayersRelative'].append({
                'jerseyNum': pl['jerseyNum'], 'r': r, 'theta': theta, 'dist_ball': dist_ball, 'angle_ball': angle_ball,
                'position': home_rosters[home_rosters.shirtNumber == int(pl['jerseyNum'])].positionGroupType.iloc[0]
            })
            if not attacks.loc[frame, 'is_home']:
                if attacks.loc[frame, 'homePlayersRelative'][-1]['position'] == 'GK':
                    attacks.loc[frame, 'GK_r'], attacks.loc[frame, 'GK_theta'] = r, theta
                    attacks.loc[frame, 'homePlayersRelative'].pop()
                elif attacks.loc[frame, 'x_flipped'] <= x_flipped:
                    inter_sgt = tangent_inter(x_flipped, y_flipped, attacks.loc[frame, 'x_flipped'], attacks.loc[frame, 'y_flipped'], 0.375, 52.5)
                    l, r = inter_sgt[0], inter_sgt[1]
                    if max(l, -9.16) <= min(r, 9.16):
                        cover_l.append(max(l, -9.16))
                        cover_r.append(min(r, 9.16))
                        
        for i in range(len(attacks.awayPlayersSmoothed[frame])):
            pl = attacks.awayPlayersSmoothed[frame][i]
            x_flipped = pl['x'] * (1 - 2 * attacks.attack_direction[frame])
            y_flipped = pl['y'] * (1 - 2 * attacks.attack_direction[frame])
            r = math.sqrt((52.5 - x_flipped)**2 + y_flipped**2)
            theta = math.atan2(y_flipped, (52.5 - x_flipped))
            delta_x, delta_y = x_flipped-attacks.x_flipped[frame], y_flipped-attacks.y_flipped[frame]
            dist_ball = math.sqrt(delta_x**2 + delta_y**2)
            angle_ball = math.atan2(delta_x, delta_y)
            attacks.loc[frame, 'awayPlayersRelative'].append({
                'jerseyNum': pl['jerseyNum'], 'r': r, 'theta': theta, 'dist_ball': dist_ball, 'angle_ball': angle_ball,
                'position': away_rosters[away_rosters.shirtNumber == int(pl['jerseyNum'])].positionGroupType.iloc[0]
            })

            if attacks.loc[frame, 'is_home']:
                if attacks.loc[frame, 'awayPlayersRelative'][-1]['position'] == 'GK':
                    attacks.loc[frame, 'GK_r'], attacks.loc[frame, 'GK_theta'] = r, theta
                    attacks.loc[frame, 'awayPlayersRelative'].pop()
                elif attacks.loc[frame, 'x_flipped'] <= x_flipped:
                    inter_sgt = tangent_inter(x_flipped, y_flipped, attacks.loc[frame, 'x_flipped'], attacks.loc[frame, 'y_flipped'], 0.375, 52.5)
                    l, r = inter_sgt[0], inter_sgt[1]
                    if max(l, -9.16) <= min(r, 9.16):
                        cover_l.append(max(l, -9.16))
                        cover_r.append(min(r, 9.16))

        # Discretize to compute the segment cover
        all_values = [-9.16, 9.16]
        for i in range(len(cover_l)):
            all_values.append(cover_l[i])
            all_values.append(cover_r[i])
        all_values = sorted(set(all_values))
        idx = {x:i for i, x in enumerate(all_values)}
        cover_pair = sorted([(idx[cover_l[i]], idx[cover_r[i]]) for i in range(len(cover_l))])

        seg, pos, covered = 0, 0, 0.0
        while seg < len(cover_l):
            covered += all_values[cover_pair[seg][0]] - all_values[pos]
            pos = cover_pair[seg][1]
            seg += 1
            while seg < len(cover_l) and cover_pair[seg][0] <= pos:
                pos = max(pos, cover_pair[seg][1])
                seg += 1
        attacks.at[frame, 'openGoal'] = (covered + (all_values[-1] - all_values[pos])) / 18.32     
                    
        dist_ball = lambda x: x['dist_ball']
        attacks.at[frame, 'homePlayersRelative'] = sorted(attacks.at[frame, 'homePlayersRelative'], key=dist_ball)
        attacks.at[frame, 'awayPlayersRelative'] = sorted(attacks.at[frame, 'awayPlayersRelative'], key=dist_ball)

    for i in range(5):
        attacks[f'DefDist{i}'] = attacks.apply(lambda row: row['awayPlayersRelative' if row.is_home else 'homePlayersRelative'][i]['dist_ball'], axis=1)
        attacks[f'OffDist{i}'] = attacks.apply(lambda row: row['awayPlayersRelative' if not row.is_home else 'homePlayersRelative'][i+1]['dist_ball'], axis=1)
        attacks[f'DefAngle{i}'] = attacks.apply(lambda row: row['awayPlayersRelative' if row.is_home else 'homePlayersRelative'][i]['angle_ball'], axis=1)
        attacks[f'OffAngle{i}'] = attacks.apply(lambda row: row['awayPlayersRelative' if not row.is_home else 'homePlayersRelative'][i+1]['angle_ball'], axis=1)

    
    attacks['competition'] = competition
    attacks['season'] = season
    attacks['game'] = game
    return attacks.drop(columns=['version', 'gameRefId', 'generatedTime', 'smoothedTime', 'homePlayers', 'homePlayersSmoothed',
       'awayPlayers', 'awayPlayersSmoothed', 'balls', 'ballsSmoothed', 'game_event', 'possession_event', 'in_play', 'x', 'y'])

competition_, season_, game_ = [], [], []
for competition in ['mls', 'pl']:
    prefix = f"pff-data/tracking/{competition}/"
    result = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
    seasons = [p['Prefix'].split('/')[-2] for p in result.get('CommonPrefixes', [])]
    print(f"Competition: {competition}")

    for season in seasons:
        season_prefix = f"{prefix}{season}/"
        result = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=season_prefix, Delimiter='/')
        games = [p['Prefix'].split('/')[-2] for p in result.get('CommonPrefixes', []) if p['Prefix'].split('/')[-2] != season]
        print(f"  Season: {season}")

        for game in games:
            key = f"{prefix}{season}/{game}/{game}.jsonl.bz2"
            competition_.append(competition)
            season_.append(season)
            game_.append(game)
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            except s3_client.exceptions.ClientError:
                print(f"    Missing file for game: {game}")
                
games = pd.DataFrame({
    'competition': competition_, 'season': season_, 'game': game_
})

pl_season = games[games.season == '2024-2025'].reset_index(drop=True)

#for j in range(19):
#    atk_train = pd.concat(
#        [process_game(pl_season.competition[i], pl_season.season[i], pl_season.game[i]) for i in range(j*20, (j+1)*20)],
#        axis=0, ignore_index=True
#    )
#    buffer = io.BytesIO()
#    atk_train.to_pickle(buffer, compression="bz2")
#    buffer.seek(0)

#    s3_client.put_object(Bucket=S3_BUCKET, Key=f'process/atk_pl_2024-2025_{j}.pkl.bz2', Body=buffer.getvalue())
#    print(f"✅ 2024-2025 Game {j} Processed")

j = 18
atk_train = pd.concat(
    [process_game(pl_season.competition[i], pl_season.season[i], pl_season.game[i]) for i in range(j*20, (j+1)*20)],
    axis=0, ignore_index=True
)
buffer = io.BytesIO()
atk_train.to_pickle(buffer, compression="bz2")
buffer.seek(0)

s3_client.put_object(Bucket=S3_BUCKET, Key=f'process/atk_pl_2024-2025_{j}.pkl.bz2', Body=buffer.getvalue())
print(f"✅ 2024-2025 Game {j} Processed")