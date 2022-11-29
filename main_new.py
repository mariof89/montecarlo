from itertools import groupby
from os import stat
import pandas as pd
from load_data import df_from_csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import stats
import datetime
from simulation import MCS

pd.options.mode.chained_assignment = None
_selected_teams = ['Juventus', 'Milan', 'Spezia']


def create_home_away_fields(df_match_complete, home_away_split_char, match_field_name) -> pd.DataFrame:
    # df_match_complete[['team_home', 'team_away']
    #                  ] = df_match_complete[match_field_name].str.split(home_away_split_char, expand=True)
    df_ret = df_match_complete.copy(deep=True)

    df_ret['team_home'] = df_ret.apply(
        lambda x: x[match_field_name].split(home_away_split_char)[0], axis=1)
    df_ret['team_away'] = df_ret.apply(
        lambda x: x[match_field_name].split(home_away_split_char)[1], axis=1)
    df_ret['team_home'] = df_ret['team_home'].str.title()
    df_ret['team_away'] = df_ret['team_away'].str.title()
    df_ret['team'] = df_ret['team'].str.title()
    return df_ret


def team_alignment(df_match_complete, home_away_split_char, match_field_name):
    # df_match_complete[['team_home', 'team_away']
    #                  ] = df_match_complete[match_field_name].str.split(home_away_split_char, expand=True)

    # , 'team_away']
    # df_match_complete['team_home'] = df_match_complete['team_home'].str.strip()
    # df_match_complete['team_away'] = df_match_complete['team_away'].str.strip()

    df_match_complete = create_home_away_fields(
        df_match_complete=df_match_complete, home_away_split_char=home_away_split_char, match_field_name=match_field_name)

    # print(df_match_complete['team_home'].nunique())
    # print(df_match_complete[['team', 'team_home', 'team_away']].head(50))

    df_match_complete[df_match_complete.team == 'Internazionale']['team'] = df_match_complete.team.replace(
        {'Internazionale': 'Inter'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Inter Milan']['team_home'] = df_match_complete.team_home.replace({
        'Inter Milan': 'Inter'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Inter Milan']['team_away'] = df_match_complete.team_away.replace({
        'Inter Milan': 'Inter'}, inplace=True)

    df_match_complete[df_match_complete.team == 'Hellas Verona']['team_home'] = df_match_complete.team_home.replace({
        'Hellas Verona': 'Verona'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Hellas Verona']['team_away'] = df_match_complete.team_away.replace({
        'Hellas Verona': 'Verona'}, inplace=True)

    df_match_complete[df_match_complete.team == 'Sportiva Salernitana']['team_home'] = df_match_complete.team_home.replace(
        {'Sportiva Salernitana': 'Salernitana'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Sportiva Salernitana']['team_away'] = df_match_complete.team_away.replace(
        {'Sportiva Salernitana': 'Salernitana'}, inplace=True)

    df_match_complete[df_match_complete.team == 'Spezia Calcio']['team_home'] = df_match_complete.team_home.replace({
        'Spezia Calcio': 'Spezia'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Spezia Calcio']['team_away'] = df_match_complete.team_away.replace({
        'Spezia Calcio': 'Spezia'}, inplace=True)

    print(df_match_complete[df_match_complete.index ==
                            2229098][['team_home', 'team_away']])
    return df_match_complete


def split_home_away(df_match_complete):
    conditions = [(df_match_complete['team_home'] == df_match_complete['team']),
                  (df_match_complete['team_away'] == df_match_complete['team'])]
    values = ['H', 'A']
    df_match_complete['home_away'] = np.select(conditions, values)
    df_match_home = df_match_complete[df_match_complete['team_home']
                                      == df_match_complete['team']]  # .groupby('game_id')['xG'].sum()
    df_match_away = df_match_complete[df_match_complete['team_away']
                                      == df_match_complete['team']]
    return {
        'home': df_match_home,
        'away': df_match_away
    }


def get_numeric_col_names_from_df(df, exceptions=None):
    _num_col_names = df.select_dtypes(include=np.number).columns
    if not exceptions:
        return list(_num_col_names)
    else:
        _num_col_names = [
            elem for elem in _num_col_names if elem not in exceptions]

    return _num_col_names


def create_history_of_matches(df_match_complete, df_match_home, df_match_away):

    _match_days = sorted(df_match_complete['matchday'].unique())
    _dict_days = {}

    _cols_grouping = ['game_id', 'team_home',
                      'team_away', 'matchday']  # , 'home_away']

    _last_days = 3
    for _day in _match_days:

        df_complete_day = df_match_complete[df_match_complete['matchday'] == _day]
        df_home_day = df_match_home[df_match_home['matchday'] == _day]
        df_away_day = df_match_away[df_match_away['matchday'] == _day]

        df_complete_day.fillna(0)
        df_home_day.fillna(0)
        df_away_day.fillna(0)

        df_complete_day_grouped = df_complete_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')
        df_home_day_grouped = df_home_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')
        df_away_day_grouped = df_away_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')

        df_complete_day_previous = df_match_complete[df_match_complete['matchday'] < _day]
        df_home_day_previous = df_match_home[df_match_home['matchday'] < _day]
        df_away_day_previous = df_match_away[df_match_away['matchday'] < _day]
        df_complete_day_previous.fillna(0)
        df_home_day_previous.fillna(0)
        df_away_day_previous.fillna(0)

        # print(df_complete_day_previous.head(10))
        # print(df_home_day_previous.shape)
        # print(df_away_day_previous.shape)

        '''
        if df_complete_day_previous.shape[0] > 0 \
            and df_home_day_previous.shape[0] > 0 \
                and df_away_day_previous.shape[0] > 0:
            df_complete_grouped_previous_match_day_complete = df_complete_day_previous.groupby(
                _cols_grouping[0:-1], as_index=False).mean().set_index('game_id')
            df_home_grouped_previous_matchday = df_home_day_previous.groupby(
                _cols_grouping[0:-1], as_index=False).mean().set_index('game_id')
            df_away_grouped_previous_matchday = df_away_day_previous.groupby(
                _cols_grouping[0:-1], as_index=False).mean().set_index('game_id')
        '''
        _numeric_col_names_complete = get_numeric_col_names_from_df(
            df_complete_day_previous, exceptions=['matchday', 'game_id'])
        _numeric_col_names_home = get_numeric_col_names_from_df(
            df_home_day_previous, exceptions=['matchday', 'game_id'])
        _numeric_col_names_away = get_numeric_col_names_from_df(
            df_away_day_previous, exceptions=['matchday', 'game_id'])

        '''
        if len(_numeric_col_names_complete) > 0:
            _numeric_col_names_complete.remove(
                'matchday')  # .remove('game_id')
        if len(_numeric_col_names_home) > 0:
            _numeric_col_names_home.remove('matchday')  # .remove('game_id')
        if len(_numeric_col_names_away) > 0:
            _numeric_col_names_away.remove('matchday')  # .remove('game_id')
        '''
        # if df_complete_day_previous.shape[0] > 0 \
        #    and df_home_day_previous.shape[0] > 0 \
        #        and df_away_day_previous.shape[0] > 0:
        df_complete_grouped_previous_match_day_complete = df_complete_day_previous.groupby(
            _cols_grouping, as_index=False)[stats._numeric_fields].sum().set_index('game_id')
        df_home_grouped_previous_matchday = df_home_day_previous.groupby(
            _cols_grouping, as_index=False)[stats._numeric_fields].sum().set_index('game_id')
        df_away_grouped_previous_matchday = df_away_day_previous.groupby(
            _cols_grouping, as_index=False)[stats._numeric_fields].sum().set_index('game_id')

        # print(df_complete_grouped_previous_match_day_complete)
        # print(df_home_grouped_previous_matchday)
        # print(df_away_grouped_previous_matchday)

        # print(df_complete_grouped_previous_match_day_complete.shape)
        # print(df_home_grouped_previous_matchday.shape)
        # print(df_away_grouped_previous_matchday.shape)
        # df_home_day_previous
        # df_home_day_previous_last_days = df_home_day_previous[]
        # df_away_day_previous_last_days = None
        # else:
        #    df_complete_grouped_previous_match_day_complete = pd.DataFrame()
        #    df_home_grouped_previous_matchday = pd.DataFrame()
        #    df_away_grouped_previous_matchday = pd.DataFrame()

        # df_home_grouped_previous_matchday['xG_previous'] = df_home_grouped_previous_matchday[df_home_grouped_previous_matchday]
        _completed_agg_key = 'completed_aggregated_previous_'
        _home_agg_key = 'home_aggregated_previous_'
        _away_agg_key = 'away_aggregated_previous_'
        _agg_key = {}

        _prev_match_days = sorted(
            df_home_grouped_previous_matchday['matchday'].unique(), reverse=True)

        '''
        *** Da modificare come vengono valutate le partite da prendere per le ultime giornate.
        Deve essere inserito un ciclo su tutte le squadre in modo tale da poter inserire la condizione sulla
        singola squadra
        '''
        for i in [3, 5]:
            if i < len(_prev_match_days):
                _prev_match_days_temp = _prev_match_days[0:i+1]
            else:
                _prev_match_days_temp = _prev_match_days

            # if df_home_grouped_previous_matchday.shape[0] > 0 and \
            #        df_away_grouped_previous_matchday.shape[0] > 0:
            '''
            if (_day == 11):
                print(_prev_match_days_temp)
                print(df_home_grouped_previous_matchday[
                    (df_home_grouped_previous_matchday['matchday'].isin(_prev_match_days_temp))][['team_home', 'team_away', 'matchday']])
                # sys.exit()
            '''
            '''
            _agg_key[_home_agg_key+f'{i}'] = df_home_grouped_previous_matchday[
                (df_home_grouped_previous_matchday['matchday'].isin(_prev_match_days_temp))].groupby(
                ['team_home']).mean().reset_index()
            _agg_key[_away_agg_key+f'{i}'] = df_away_grouped_previous_matchday[
                (df_away_grouped_previous_matchday['matchday'].isin(_prev_match_days_temp))].groupby(
                ['team_away']).mean().reset_index()
            '''
            # print(df_home_grouped_previous_matchday)
            # print(df_away_grouped_previous_matchday)
            # _agg_key[_home_agg_key+f'{i}'] = df_home_grouped_previous_matchday.sort_values(
            #    by=['team_home', 'matchday'], ascending=False).groupby(['team_home']).head(i).mean().reset_index()  # .head(i).groupby(['team_home']).mean().reset_index()  # .head[i]  # .head[i].reset_index()
            '''
            if _day == 5:
                _df = df_home_grouped_previous_matchday.sort_values(
                    by=['team_home', 'matchday'], ascending=False)  # [['team_home', 'matchday', 'xG']]
                print(_df[['team_home', 'matchday', 'xG']])
                _df2 = _df.groupby(['team_home']).head(
                    i).groupby('team_home')[stats._numeric_fields].mean().reset_index()  # .mean()
                # .groupby(level=[0, 1]).head(2).reset_index()

                print(_df2)
                sys.exit()
            '''
            _agg_key[_home_agg_key+f'{i}'] = df_home_grouped_previous_matchday.sort_values(
                by=['team_home', 'matchday'], ascending=False).groupby(['team_home']).head(
                    i).groupby('team_home')[stats._numeric_fields].mean().reset_index()

            # print(_agg_key[_home_agg_key+f'{i}'])
            # sys.exit()
            # _agg_key[_home_agg_key+f'{i}']['xG_last_3'] = _agg_key[_home_agg_key +
            #                                                       f'{i}'].groupby(['team_home']).apply(lambda x: x.sort_values(['matchday'], ascending=False))  # .apply(lambda v: v.mean())  # .head(i)['xG'].mean()
            '''
            if _day == 11:
                _df = df_home_grouped_previous_matchday.sort_values(
                    by=['team_home', 'matchday'], ascending=False)  # [['team_home', 'matchday', 'xG']]
                print(_df[['team_home', 'matchday', 'xG']])
                print(_agg_key[_home_agg_key+f'{i}'][['team_home', 'xG']])
                # print(
                #    _agg_key[_home_agg_key+f'{i}'].groupby('team_home').head(i).mean()['xG'])
                # print(
                #    _agg_key[_home_agg_key+f'{i}'][['team_home', 'matchday', 'xG', 'xG_last_3']])
                sys.exit()
            # print(_agg_key[_home_agg_key+f'{i}'].columns)
            '''
            '''
            _df = {'team_home': list(
                _agg_key[_home_agg_key+f'{i}']['team_home'].unique())}
            for _team in _df['team_home']:
                for _m in stats._numeric_fields:
                    _df[_m] = _agg_key[_home_agg_key +
                                       f'{i}'][_agg_key[_home_agg_key+f'{i}'].team_home == _team].head(i)[_m].mean()

            _agg_key[_home_agg_key+f'{i}'] = pd.DataFrame(_df)
            print(_agg_key[_home_agg_key+f'{i}'])
            '''

            _agg_key[_away_agg_key+f'{i}'] = df_away_grouped_previous_matchday.sort_values(
                by=['matchday'], ascending=False).groupby(['team_away']).head(
                    i).groupby('team_away')[stats._numeric_fields].mean().reset_index()
            '''
            if (_day == 11):
                print(_agg_key['home_aggregated_previous_3']
                      [['team_home', 'matchday', 'xG']])
                # sys.exit()
            '''
            '''
            _agg_key[_home_agg_key+f'{i}'] = df_home_grouped_previous_matchday[
                (df_home_grouped_previous_matchday['matchday'] >= (_day - i)) &
                (df_home_grouped_previous_matchday['matchday'] < (_day - 1))].groupby(
                ['team_home']).mean().reset_index()
            _agg_key[_away_agg_key+f'{i}'] = df_away_grouped_previous_matchday[
                (df_away_grouped_previous_matchday['matchday'] >= (_day - i)) &
                (df_away_grouped_previous_matchday['matchday'] < (_day - 1))].groupby(
                ['team_away']).mean().reset_index()
            '''
            # else:
            #    _agg_key[_home_agg_key+f'{i}'] = pd.DataFrame(),
            #    _agg_key[_away_agg_key+f'{i}'] = pd.DataFrame()
        '''
        for m in stats._attacking_metrics:
            _m_rank = f'{m}_rank'
            if '_rank' not in m:
                df_complete_day[_m_rank] = 100 * \
                    df_complete_day[m].rank(pct=True)
                df_home_day[_m_rank] = 100*df_home_day[m].rank(pct=True)
                df_away_day[_m_rank] = 100*df_away_day[m].rank(pct=True)

                df_complete_day_grouped[_m_rank] = 100 * \
                    df_complete_day_grouped[m].rank(pct=True)
                df_home_day_grouped[_m_rank] = 100 * \
                    df_home_day_grouped[m].rank(pct=True)
                df_away_day_grouped[_m_rank] = 100 * \
                    df_away_day_grouped[m].rank(pct=True)

                if _day > 2:
                    df_complete_grouped_previous_match_day_complete[_m_rank] = 100 * \
                        df_complete_grouped_previous_match_day_complete[m].rank(
                            pct=True)
                    df_home_grouped_previous_matchday[_m_rank] = 100 * \
                        df_home_grouped_previous_matchday[m].rank(pct=True)
                    df_away_grouped_previous_matchday[_m_rank] = 100 * \
                        df_away_grouped_previous_matchday[m].rank(pct=True)
        '''
        _dict_day = {
            'complete': df_complete_day,
            'complete_aggregated': df_complete_day_grouped,
            'complete_aggregated_previous': df_complete_grouped_previous_match_day_complete,
            'home': df_home_day,
            'home_aggregated': df_home_day_grouped,
            'home_aggregated_previous': df_home_grouped_previous_matchday,
            'away': df_away_day,
            'away_aggregated': df_away_day_grouped,
            'away_aggregated_previous': df_away_grouped_previous_matchday
        }

        _dataframes_current_previous = []
        _metrics_to_use = ['xG', 'xT', 'xA']
        for k, v in _agg_key.items():
            _dict_day[k] = v

        # print(_dict_day['complete_aggregated'])
        _current_with_previous_data = pd.DataFrame()
        _current_with_previous_data['game_id'] = list(
            _dict_day['complete_aggregated'].index)
        _current_with_previous_data['team_home'] = list(
            _dict_day['complete_aggregated']['team_home'])
        _current_with_previous_data['team_away'] = list(
            _dict_day['complete_aggregated']['team_away'])
        _current_with_previous_data['matchday'] = list(
            _dict_day['complete_aggregated']['matchday'])

        if _day > 0:
            print(f"Day: {_day}")
            # print(_dict_day['home_aggregated_previous_5'][['team_home', 'xG']])
            # print(_dict_day['home_aggregated_previous_3'])
            # if _dict_day['home_aggregated_previous_3']:
            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data[
                    ['game_id', 'matchday', 'team_home', 'team_away']],
                # 'indicator_away'
                # [['xG']],
                right=_dict_day['home_aggregated_previous_3'][stats._numeric_fields],
                left_on=_current_with_previous_data['team_home'],
                right_on=_dict_day['home_aggregated_previous_3']['team_home'],
                how='inner').drop('key_0', axis=1)

            # if _dict_day['home_aggregated_previous_5']:
            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data[
                    ['game_id', 'matchday', 'team_home', 'team_away'] + stats._numeric_fields],
                # 'indicator_away'
                right=_dict_day['home_aggregated_previous_5'][stats._numeric_fields],
                left_on=_current_with_previous_data['team_home'],
                right_on=_dict_day['home_aggregated_previous_5']['team_home'],
                how='inner').drop('key_0', axis=1)

            _dict_for_renaming = {}
            '''
            _dict_for_renaming = {
                'xG_x': 'xG_home_previous_3',
                'xG_y': 'xG_home_previous_5'}
            '''
            _cols_for_merging = []

            # for _m in _metrics_to_use:
            for _c in list(_current_with_previous_data.columns):

                # print(_c[len(_c)-2:len(_c)])
                if _c.endswith('_x'):  # _c[len(_c)-2:len(_c)] == '_x':
                    _k = _c.replace('_x', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_home_previous_3'
                elif _c.endswith('_y'):  # _c[len(_c)-2:len(_c)] == '_y':
                    _k = _c.replace('_y', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_home_previous_5'

            _current_with_previous_data.rename(
                columns=_dict_for_renaming, inplace=True)

            # print(_current_with_previous_data.head(10))

            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data,  # [
                # ['game_id', 'matchday', 'team_home', 'team_away'] + _cols_for_merging],
                # 'indicator_away'
                right=_dict_day['away_aggregated_previous_3'][stats._numeric_fields],
                left_on=_current_with_previous_data['team_away'],
                right_on=_dict_day['away_aggregated_previous_3']['team_away'],
                how='inner').drop('key_0', axis=1)
            if _day == 20:
                print(_current_with_previous_data[[
                      'game_id', 'matchday', 'team_home', 'team_away']])
            # if _dict_day['away_aggregated_previous_5']:
            _current_with_previous_data = pd.merge(
                _current_with_previous_data,  # [
                # ['game_id', 'matchday', 'team_home', 'team_away'] + _cols_for_merging + _metrics_to_use],
                # 'xG_home_previous_3', 'xG_home_previous_5', 'xG']],
                # 'indicator_away'
                _dict_day['away_aggregated_previous_5'][stats._numeric_fields],
                left_on=_current_with_previous_data['team_away'],
                right_on=_dict_day['away_aggregated_previous_5']['team_away'],
                how='inner').drop('key_0', axis=1)

            for _c in list(_current_with_previous_data.columns):

                # print(_c[len(_c)-2:len(_c)])
                if _c.endswith('_x'):  # _c[len(_c)-2:len(_c)] == '_x':
                    _k = _c.replace('_x', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_away_previous_3'
                elif _c.endswith('_y'):  # _c[len(_c)-2:len(_c)] == '_y':
                    _k = _c.replace('_y', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_away_previous_5'

            _current_with_previous_data.rename(
                columns=_dict_for_renaming, inplace=True)

            # _dataframes_current_previous.append(_current_with_previous_data)

            # _dict_day[_day] = _current_with_previous_data

            # if 'xG_home_previous_3' in list(_current_with_previous_data.columns):
            # print(_current_with_previous_data.head(10)
            #      [['team_home', 'team_away', 'xG_home_previous_3', 'xG_home_previous_5', 'xG_away_aggregated_previous_3', 'xG_away_aggregated_previous_5']])
            '''
            for k, v in _agg_key.items():
                print(k)
                if 'home' in k and '_previous_' in k:
                    _k = int(k[-1])

                    _current_with_previous_data = pd.merge(
                        _current_with_previous_data[
                            ['game_id', 'matchday', 'team_home', 'team_away']],
                        _dict_day[k][['xG']],  # 'indicator_away'
                        left_on=_current_with_previous_data['team_home'],
                        right_on=_dict_day[k]['team_home'],
                        how='inner').drop('key_0', axis=1)  # .set_index('game_id')

                    _dict_for_renaming = {
                        'xG': 'xG_'+k}

                    _current_with_previous_data.rename(
                        columns=_dict_for_renaming, inplace=True)


                    _current_with_previous_data = pd.merge(
                        _current_with_previous_data[
                            ['game_id', 'matchday', 'team_home', 'team_away']],
                        _dict_day[k][['xG']],  # 'indicator_away'
                        left_on=_current_with_previous_data['team_home'],
                        right_on=_dict_day[k]['team_home'],
                        how='inner').drop('key_0', axis=1)

                    # print(_current_with_previous_data)


            for m in ['xG']:  # get_numeric_col_names_from_df(
                # _dict_day['complete_aggregated'], exceptions=['game_id', 'matchday', 'opta_id']):

                for k, v in _agg_key.items():
                    print(v)
                    if '_previous_' in k:
                        _k = int(k[-1])
                        _current_with_previous_data[f'{m}_home_previous_{_k}'] = \
                            _dict_day['complete_aggregated'][
                                _dict_day['complete_aggregated']['team_home'] == v.team_home]
            '''
        # print(_current_with_previous_data)
        _dict_day['dataframe_for_simulation'] = _current_with_previous_data

        _dict_days[_day] = _dict_day

    return _dict_days


if __name__ == '__main__':
    path_soccerment = '../datasets/soccerment_serieA_2021-22_giocatori.csv'
    path_sics = '../datasets/sics_serieA_2021-22.csv'
    path_skillcorner = '../datasets/skillcorner_serieA_2021-22.csv'
    path_playerid = '../datasets/player_id_serieA_2021-22.csv'

    path_matches = '../datasets/dati_completi_serieA_2021-22_partite.csv'

    path_ipo = '../datasets/SICS_SerieA_2021-22_OffensiveIndexTimetable.csv'

    '''
        Load data related to players
    '''
    df_match_complete = df_from_csv(path=path_matches)  # .set_index('game_id')
    df_match_complete = df_match_complete.drop_duplicates(
        ['game_id', 'opta_id'], keep='first')

    df_ipo = df_from_csv(path=path_ipo, delimiter=';')

    # df_ipo = team_alignment(
    #    df_ipo, home_away_split_char='-', match_field_name='match_name')  # .set_index(['team_home', 'team_away', 'team'])

    # .set_index(['team_home', 'team_away'])
    # df_ipo_by_match = df_ipo.groupby(['team_home', 'team_away', 'team'])[
    #    'weight'].sum().reset_index()

    # df_ipo_home_away = split_home_away(df_ipo)
    # df_ipo_home = df_ipo_home_away['home']

    # df_ipo_away = df_ipo_home_away['away']

    # df_ipo_home_by_match = df_ipo_home_away['home'].groupby(
    #    ['matchDay', 'team_home', 'team_away', 'team']).sum().reset_index()
    # df_ipo_away_by_match = df_ipo_home_away['away'].groupby(
    #    ['matchDay', 'team_home', 'team_away', 'team']).sum().reset_index()

    # print(df_ipo_home_by_match)
    # print(df_ipo_away_by_match)
    '''
        Viene effettuato un alignment per normalizzare i nomi delle squadre
    '''
    df_match_complete = team_alignment(
        df_match_complete, home_away_split_char=' v ', match_field_name='Match')

    '''
        Viene raggruppato il dataframe in modo tale da avere una riga x ogni partita: 380 righe
    '''
    df_match_grouped = df_match_complete.groupby(
        ['game_id', 'team', 'opponent_team']).sum()

    '''
        A partire dal DF originario, si creano 2 dataframe distinti,
        uno per le partite in casa e uno per quelle in trasferta
    '''
    dfs_ha = split_home_away(df_match_complete)

    df_match_home = dfs_ha['home']
    df_match_away = dfs_ha['away']

    # df_match_home['ipo_home'] = df_match_home.index.map(df_ipo_by_match.weight)
    # df_match_away['ipo_away'] = df_match_away.index.map(df_ipo_by_match.weight)

    # print(df_match_home[df_match_home.matchday == 1][['full_name']])
    # sys.exit()

    '''
        Vengono creati tanti dataframe per avere la situazione alla giornata precedente.
        Ad esempio, la giornata 10, avra un dizionario così composto:

        {
            'complete': DF contenente tutto il dettaglio dei giocatori di quella giornata,
            'complete_aggregated': df_complete_day_grouped,
            'home': df_home_day,
            'home_aggregated': df_home_day_grouped,
            'home_aggregated_previous': df_home_grouped_previous_matchday,
            'away': df_away_day,
            'away_aggregated': df_away_day_grouped,
            'away_aggregated_previous': df_away_grouped_previous_matchday
        }

    '''
    dataframe_dict = create_history_of_matches(
        df_match_complete, df_match_home, df_match_away)

    _df_final_for_simulation = pd.concat([v['dataframe_for_simulation']
                                          for k, v in dataframe_dict.items()])
    '''
    _dict_for_renaming = {
        'xG_home_previous_3': 'indicator_home',
        'xG_away_aggregated_previous_3': 'indicator_away'
    }
    '''
    # _df_final_for_simulation.rename(columns=_dict_for_renaming, inplace=True)

    for p in _df_final_for_simulation.columns:
        print(p)

    _metrics_to_use = {
        'metrics': ['xG', 'xT', 'xA'],
        'weights': [1., 1., 1.]
    }

    _metrics_selection = {}

    for ha in ['home', 'away']:
        _metrics_selection[ha] = {}
        for i in [3, 5]:
            _metrics_selection[ha][i] = []
            for _m in _metrics_to_use['metrics']:
                _cnt = f'{_m}_{ha}_previous_{i}'
                _metrics_selection[ha][i].append(_cnt)
                # _metrics_selection[ha].append(_cnt)
                # l.append(f'{_m}_{ha}_previous_{i}')

    _df_final_for_simulation = _df_final_for_simulation.drop_duplicates(
        ['game_id', 'matchday'])
    print(_df_final_for_simulation[_df_final_for_simulation['matchday'] == 20][
          ['game_id', 'matchday', 'team_home', 'team_away', 'xG_home_previous_3', 'xG_away_previous_3', 'xG_home_previous_5']])

    print("\n\n")
    print(_df_final_for_simulation[[
          'team_home', 'team_away', 'matchday', 'xG_home_previous_3']])
    # for k, v in _metrics_selection.items():
    #    print(_metrics_selection[k])

    _w = [.5, .5]
    _st = ['xG_home_previous_3', 'xT_home_previous_3']

    for ha, v in _metrics_selection.items():
        for k, v1 in _metrics_selection[ha].items():
            if k == 3:
                _df_final_for_simulation[f'indicator_{ha}'] = _df_final_for_simulation.apply(
                    lambda x: np.mean(
                        x[_metrics_selection[ha][k]]), axis=1)

    # lambda x: np.average(x[['xG_home_previous_3', 'xT_home_previous_3']], weights=np.array([1., 2.], dtype=np.float)), axis=1)
    # weights=_metrics_to_use['weights']),
    # axis=1)

    # print(_df_final_for_simulation[_df_final_for_simulation['matchday'] == 11][
    #      ['game_id', 'matchday', 'team_home', 'team_away', 'xG_home_previous_3', 'xT_home_previous_3', 'xG_home_previous_3', 'indicator_home', 'indicator_away']])

    _df_final_for_simulation.to_csv('metrics.csv')
    df_simulation = MCS(
        _df_final_for_simulation, considered_metrics=['xG'], num_simulations=20000).start_sim()

    df_simulation.to_csv('Simulated_matches.csv')
    sys.exit()
    print(dataframe_dict[10]['complete_aggregated']
          [['team_home', 'team_away', 'xG']])
    print(dataframe_dict[10]['home_aggregated_previous_5']
          [['team_home', 'xG']])
    print(dataframe_dict[10]['away_aggregated_previous_5']
          [['team_away', 'xG']])
    sys.exit()
    _df_to_concat_previous = []
    _df_to_concat_current = []

    '''
    print(dataframe_dict[5]['home_aggregated_previous'].head(25)
          [['team_home', 'team_away', 'matchday', 'xG']])
    print(dataframe_dict[5]['away_aggregated_previous'].head(25)
          [['team_home', 'team_away', 'matchday', 'xG']])
    '''
    # Con questo ciclo viene creato un dataframe di 380 righe (tutto il campionato). Ma xG di riferimento (o altra metrica)
    # è relativo a qulla partita specifica
    for k, v in dataframe_dict.items():

        _completed_agg_key = 'completed_aggregated_previous_'
        _home_agg_key = 'home_aggregated_previous_'
        _away_agg_key = 'away_aggregated_previous_'
        if v['home_aggregated_previous'] is None or v['away_aggregated_previous'] is None:
            continue

        for i in [3, 5]:
            # v[_completed_agg_key+f'{i}'] = v['complete_aggregated'][
            #    (v['complete_aggregated']['matchday'] > k - i) & (v['complete_aggregated']['matchday'] < k - 1)].groupby(
            #    ['team_home']).mean()
            v[_completed_agg_key+f'{i}'] = v['complete_aggregated'][
                (v['complete_aggregated']['matchday'] > k - i) & (v['complete_aggregated']['matchday'] < k - 1)].groupby(
                ['team']).mean().reset_index()
            v[_home_agg_key+f'{i}'] = v['home_aggregated_previous'][
                (v['home_aggregated_previous']['matchday'] > k - i) & (v['home_aggregated_previous']['matchday'] < k - 1)].groupby(
                ['team_home']).mean().reset_index()
            v[_away_agg_key+f'{i}'] = v['away_aggregated_previous'][
                (v['away_aggregated_previous']['matchday'] > k - i) & (v['away_aggregated_previous']['matchday'] < k - 1)].groupby(
                ['team_away']).mean().reset_index()
        # v[_away_agg_key] = v['away_aggregated_previous'].groupby(
        #    ['game_id', 'team_home', 'team_away']).mean()

        # v[_home_agg_key]['indicator_home'] = v['home_aggregated_previous'].apply(
        #    lambda x: np.mean(x[stats_to_consider]), axis=1)
        # v[_away_agg_key]['indicator_away'] = v['away_aggregated_previous'].apply(
        #    lambda x: np.mean(x[stats_to_consider]), axis=1)

        result_previous = pd.merge(
            # [['team_home', 'xG', 'Gol', 'matchday']],
            v['home_aggregated_previous'][[
                'team_home', 'matchday']],  # 'indicator_home'
            # [['team_away', 'xG', 'Gol']],
            v['away_aggregated_previous'][['team_away']],  # 'indicator_away'
            left_on=v['home_aggregated_previous'].index,
            right_on=v['away_aggregated_previous'].index,
            how='inner')

        result_current = pd.merge(
            # [['team_home', 'xG', 'Gol', 'matchday']],
            v['home_aggregated'][['team_home', 'matchday']],
            # [['team_away', 'xG', 'Gol']],
            v['away_aggregated'][['team_away']],
            left_on=v['home_aggregated'].index,
            right_on=v['away_aggregated'].index,
            how='inner')

        print(f'result_previous.shape: {result_previous.shape}')
        print(f'result_current.shape: {result_current.shape}')
        _df_to_concat_previous.append(result_previous)
        _df_to_concat_current.append(result_current)

    result_completed_previous = pd.concat(_df_to_concat_previous)
    result_completed_current = pd.concat(_df_to_concat_current)

    _dict_for_renaming = {
        'key_0': 'game_id', 'indicator_x': 'indicator_home', 'indicator_y': 'indicator_away'}

    for _c in result_completed_previous.columns:
        if '_home_x' in _c:
            _dict_for_renaming[_c] = _c.replace('_home_x', '_home')
        elif '_away_x' in _c:
            _dict_for_renaming[_c] = _c.replace('_away_x', '_away')
        elif '_home' not in _c and _c.endswith('_x'):
            _dict_for_renaming[_c] = _c.replace('_x', '_home')
        elif '_away' not in _c and _c.endswith('_y'):
            _dict_for_renaming[_c] = _c.replace('_y', '_away')

    for _c in result_completed_current.columns:
        if '_home_x' in _c:
            _dict_for_renaming[_c] = _c.replace('_home_x', '_home')
        elif '_away_x' in _c:
            _dict_for_renaming[_c] = _c.replace('_away_x', '_away')
        elif '_home' not in _c and _c.endswith('_x'):
            _dict_for_renaming[_c] = _c.replace('_x', '_home')
        elif '_away' not in _c and _c.endswith('_y'):
            _dict_for_renaming[_c] = _c.replace('_y', '_away')

    result_completed_previous.rename(
        columns=_dict_for_renaming,
        inplace=True)

    result_completed_current.rename(
        columns=_dict_for_renaming,
        inplace=True)

    # print(result_completed_previous.head(5)['indicator_home'])

    # Quello da fare è andare a fare la media delle metriche delle giornate precenti. Fare una media pesata e creare un
    # indicatore unico

    print(dataframe_dict[5]['home_aggregated_previous_5']
          [['team_home', 'xG']])
    print(dataframe_dict[5]['away_aggregated_previous_5']
          [['team_away', 'xG']])
    sys.exit()
    stats_to_consider = ['xG']  # , 'xT', 'xA']
    weights = [1, .5, .5]

    for k, v in dataframe_dict.items():
        _df_home_previous = v['home_aggregated_previous']
        _df_away_previous = v['away_aggregated_previous']
        _df_h_prev_last_days = None

        # print(_df_home_previous)
        if not _df_home_previous is None and not _df_away_previous is None:

            _df_h_prev_last_days = _df_home_previous[
                (_df_home_previous['matchday'] < k) &
                (_df_home_previous['matchday'] > k-5)].groupby('team_home')[stats_to_consider].mean()

            _df_a_prev_last_days = _df_home_previous[
                (_df_away_previous['matchday'] < k) &
                (_df_away_previous['matchday'] > k-5)].groupby('team_away')[stats_to_consider].mean()

            _df_h_prev_last_days['indicator_home'] = _df_h_prev_last_days.apply(
                lambda x: np.average(x[stats_to_consider], weights=weights), axis=1)

            _df_a_prev_last_days['indicator_away'] = _df_a_prev_last_days.apply(
                lambda x: np.average(x[stats_to_consider], weights=weights), axis=1)

            if not _df_h_prev_last_days is None and not _df_a_prev_last_days is None:
                print(f'day: {k} - Shape: {_df_h_prev_last_days.shape}')
                print(f'day: {k} - Shape: {_df_a_prev_last_days.shape}')

            print(_df_h_prev_last_days.head(10))
        # print(f'day: {k} - Shape: {None if not _df_h_prev_last_days else _df_h_prev_last_days.shape }')
        # if _df_h_prev_last_days.shape[0] > 0

    '''
        TO-DO:  Per una data partita, calcolare l'indicatore che della media di home/away
        '''
    # print(k)
    # if (v['home_aggregated_previous'] is None):
    #    continue

    '''
        v['home_aggregated_previous']['indicator_home'] = v['home_aggregated_previous'].apply(
            lambda x: np.mean(x[stats_to_consider]), axis=1)
        v['away_aggregated_previous']['indicator_away'] = v['away_aggregated_previous'].apply(
            lambda x: np.mean(x[stats_to_consider]), axis=1)
        '''
    # print(k)
    # print(f"Game ID: {v['home_aggregated_previous']['game_id']}")
    # print(f'Mean: {_mean}')
    # _wm = v['home_aggregated_previous'].apply(
    #    lambda x: np.average(x[stats_to_consider], weights=weights), axis=1)
    # print(f'{k}: {_wm}')
    '''
        result = pd.merge(
            v['home_aggregated'][['team_home', 'Gol', 'matchday'] + \
                stats_to_consider + ['indicator']],
            v['away_aggregated'][['team_away', 'Gol'] + \
                stats_to_consider + ['indicator']],
            left_on=v['home_aggregated'].index,
            right_on=v['away_aggregated'].index,
            how='inner')
        '''
    '''
        def wm(x): return np.average(
            x[stats_to_consider], weights=weights)
        v['home_aggregated']['indicator'] = (
            (v['home_aggregated'][stats_to_consider[0]]*weights[0]) + (v['home_aggregated'][stats_to_consider[1]]*weights[1]) + (v['home_aggregated'][stats_to_consider[2]]*weights[2]))/sum(weights)
        v['away_aggregated']['indicator'] = (
            (v['away_aggregated'][stats_to_consider[0]]*weights[0]) + (v['away_aggregated'][stats_to_consider[1]]*weights[1]) + (v['away_aggregated'][stats_to_consider[2]]*weights[2]))/sum(weights)
        result = pd.merge(
            v['home_aggregated'][['team_home', 'Gol', 'matchday'] +
                                 stats_to_consider + ['indicator']],
            v['away_aggregated'][['team_away', 'Gol'] +
                                 stats_to_consider + ['indicator']],
            left_on=v['home_aggregated'].index,
            right_on=v['away_aggregated'].index,
            how='inner')
        '''
    # _df_to_concat.append(result)

    # result_completed = pd.concat(_df_to_concat)

    '''
    result_completed.rename(columns={
        'key_0': 'game_id',
        'xG_x': 'xG_home',
        'matchday_x': 'matchday',
        'xG_y': 'xG_away',
        'xA_x': 'xA_home',
        'xA_y': 'xA_away',
        'indicator_x': 'indicator_home',
        'indicator_y': 'indicator_away',
    }, inplace=True)
    '''

    df_simulation = MCS(
        result_completed_current, considered_metrics=stats_to_consider).start_sim()
    _filename = '_'.join(stats_to_consider)
    _dt = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    df_simulation.to_csv(f'simulated_matches_{_filename}_{_dt}.csv')

    print(df_simulation.head(5)[['p_home_win', 'p_draw', 'p_away_win']])

    # for k, v in dataframe_dict.keys():
    #    v['home_aggregated'].merge(v['away_aggregated'])

    # print(dataframe_dict[1]['home_aggregated'].shape)
    print(dataframe_dict[1]['home_aggregated'].head(10)
          [['team_home', 'xG']])

    # print(dataframe_dict[1]['away_aggregated'].shape)
    print(dataframe_dict[1]['away_aggregated'].head(10)
          [['team_away', 'xG']])

    print(dataframe_dict[3]['home_aggregated_previous'].head(
        10)[['team_home', 'xG']])
    print(dataframe_dict[3]['away_aggregated_previous'].head(
        10)[['team_away', 'xG']])

    print(dataframe_dict[1]['complete_aggregated'][dataframe_dict[1]
          ['complete_aggregated'].home_away == 'H'].shape)
