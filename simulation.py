import sys
import math
import time
import pandas as pd
import numpy as np
from prettytable import PrettyTable


class MCS:
    def __init__(self, df_matches, considered_metrics, num_simulations=20000):
        self.df_matches = df_matches
        self.num_simulations = num_simulations

        self.p_home_win = 0
        self.p_away_win = 0
        self.p_draw = 0

        self.home_xPts = 0
        self.away_xPts = 0

        self.considered_metrics = considered_metrics

    def single_simulation(self, home_indicator, away_indicator):
        target_home_goals_scored = np.random.poisson(home_indicator)
        target_away_goals_scored = np.random.poisson(away_indicator)

        home_win = 0
        away_win = 0
        draw = 0
        margin = 0

        # if more goals for home team => home team wins
        if target_home_goals_scored > target_away_goals_scored:
            home_win = 1
            margin = target_home_goals_scored - target_away_goals_scored
        # if more goals for away team => away team wins
        elif target_home_goals_scored < target_away_goals_scored:
            away_win = 1
            margin = target_away_goals_scored - target_home_goals_scored
        else:
            draw = 1
            margin = target_away_goals_scored - target_home_goals_scored

        return {
            'home_win': home_win,
            'away_win': away_win,
            'draw': draw,
            'margin': margin,
            'home_goals_scored': target_home_goals_scored,
            'away_goals_scored': target_away_goals_scored
        }

    def simulate_match(self, id_match, home_info, away_info, num_simulations):
        team_home_name = home_info['team']
        team_home_indicator = home_info['indicator']

        team_away_name = away_info['team']
        team_away_indicator = away_info['indicator']

        print("********************")
        print("*                  *")
        print("* SIMULATION TABLE *")
        print("*                  *")
        print("********************")
        count_home_wins = 0
        count_home_loss = 0
        count_away_wins = 0
        count_away_loss = 0
        count_draws = 0
        score_mat = []
        tot_sim_time = 0
        sim_table = PrettyTable(["SIMULATION #", "SIMULATION TIME (s)", team_home_name,
                                 team_away_name, "HOME WIN", "AWAY WIN", "DRAW", "SCORE MARGIN"])

        for i in range(self.num_simulations):
            start_time = time.time()

            simulated_match = self.single_simulation(
                home_indicator=team_home_indicator, away_indicator=team_away_indicator)

            home_win = simulated_match['home_win']
            away_win = simulated_match['away_win']
            draw = simulated_match['draw']
            margin = simulated_match['margin']
            home_goals_scored = simulated_match['home_goals_scored']
            away_goals_scored = simulated_match['away_goals_scored']

            # print(f"home_goals_scored: {home_goals_scored}")
            # print(f"away_goals_scored: {away_goals_scored}")
            if home_win > 0:
                count_home_wins += 1
                count_away_loss += 1
            elif away_win > 0:
                count_away_wins += 1
                count_home_loss += 1
            else:
                count_draws += 1
            score_mat.append((home_goals_scored, away_goals_scored))

            end_time = time.time()

            tot_sim_time += round((end_time - start_time), 5)

            sim_table.add_row([i+1, round((end_time - start_time), 5), home_goals_scored,
                              away_goals_scored, home_win, away_win, draw, margin])

        p_home_win = round((count_home_wins/num_simulations * 100), 2)
        p_away_win = round((count_away_wins/num_simulations * 100), 2)
        p_draw = round((count_draws/num_simulations * 100), 2)
        print("home win: "+str(p_home_win))
        print("Away win: "+str(p_away_win))
        print("Draw: "+str(p_draw))

        return {
            'p_home_win': float(p_home_win),
            'p_away_win': float(p_away_win),
            'p_draw': float(p_draw),
            'count_home_outcome': {'wins': count_home_wins, 'loss': count_home_loss, 'draws': count_draws},
            'count_away_outcome': {'wins': count_away_wins, 'loss': count_away_loss, 'draws': count_draws},
            'score_matrix': score_mat,
            'total_simulation_time': tot_sim_time
        }

    def build_result_occurrences(self, score_matrix, max_score=5):
        total_scores = len(score_matrix)
        assemble_scores = [
            [0 for x in range(max_score)] for y in range(max_score)]

        for i in range(total_scores):
            if score_matrix[i][0] >= max_score or score_matrix[i][1] >= max_score:
                continue
            _home_score = score_matrix[i][0]
            _away_score = score_matrix[i][1]
            assemble_scores[_home_score][_away_score] += 1
            '''
            try:
                assemble_scores[_home_score][_away_score] += 1
            except:
                print(f"Home Score: {_home_score}")
                print(f"Away Score: {_away_score}")
                print(assemble_scores)

            for j in range(max_score):
                for k in range(max_score):
                    if score_matrix[i][0] == j and score_matrix[i][1] == k:
                        assemble_scores[j][k] += 1
        '''

        return {
            'result_occurrences': assemble_scores
        }

    def build_score_matrix_probability(self, score_matrix_occurrences, num_simulations, round_digits=3):
        print(score_matrix_occurrences)
        print(f"Len Score matrix: {len(score_matrix_occurrences)}")

        home_score_occurrence_probability = [
            0 for x in range(len(score_matrix_occurrences))]
        away_score_occurrence_probability = [
            0 for x in range(len(score_matrix_occurrences))]
        score_matrix_prob = [
            [0 for x in range(len(score_matrix_occurrences))] for y in range(len(score_matrix_occurrences))]

        for _home_index in range(len(score_matrix_occurrences)):
            # home_score_occurrence_probability.append(
            #    round(
            #        sum(score_matrix_occurrences[_home_index]/num_simulations), round_digits)
            # )
            for _away_index in range(len(score_matrix_occurrences)):
                score_matrix_prob[_home_index][_away_index] = round(
                    score_matrix_occurrences[_home_index][_away_index]/num_simulations, round_digits)

                home_score_occurrence_probability[_home_index] += round(
                    score_matrix_occurrences[_home_index][_away_index]/num_simulations, round_digits)
                away_score_occurrence_probability[_away_index] += round(
                    score_matrix_occurrences[_home_index][_away_index]/num_simulations, round_digits)

        return {
            'home_score_occurrence_probability': home_score_occurrence_probability,
            'away_score_occurrence_probability': away_score_occurrence_probability,
            'score_matrix_probability': score_matrix_prob
        }

    def start_sim(self):
        print("*** START SIMULATION ***")
        game_ids = []
        team_homes = []
        team_aways = []
        match_days = []
        p_home_wins = []
        p_away_wins = []
        p_draws = []
        p_scoring_home = []
        p_scoring_away = []

        for i in range(0, len(self.df_matches)):

            game_id = self.df_matches.iloc[i]['game_id']
            team_home = self.df_matches.iloc[i]['team_home']
            team_away = self.df_matches.iloc[i]['team_away']
            matchday = self.df_matches.iloc[i]['matchday']
            home_team = self.df_matches.iloc[i]['team_home']
            home_team_indicator = self.df_matches.iloc[i]['indicator_home']

            game_id = self.df_matches.iloc[i]['game_id']
            away_team = self.df_matches.iloc[i]['team_away']
            away_team_indicator = self.df_matches.iloc[i]['indicator_away']

            home_info = {
                'team': home_team,
                'indicator': away_team_indicator
            }

            away_info = {
                'team': away_team,
                'indicator': away_team_indicator
            }

            print("* Game #", i+1, "*")
            print("* Home team:", home_team)
            print("* Away team:", away_team)
            print("* Home team Indicator:",
                  home_team_indicator)
            print("* Away team Indicator:",
                  away_team_indicator)

            simulated_match = self.simulate_match(
                id_match=game_id,
                home_info=home_info,
                away_info=away_info,
                num_simulations=self.num_simulations)

            total_simulation_time = simulated_match['total_simulation_time']
            count_home_outcome = simulated_match['count_home_outcome']
            count_away_outcome = simulated_match['count_away_outcome']
            p_home_win = simulated_match['p_home_win']
            p_away_win = simulated_match['p_away_win']
            p_draw = simulated_match['p_draw']
            score_matrix = simulated_match['score_matrix']

            # print the simulation statistics
            print("*************")
            print("*           *")
            print("* SIM STATS *")
            print("*           *")
            print("*************")
            sim_table_stats = PrettyTable(
                ["Total # of sims", "Total time (s) for sims", "HOME WINS", "AWAY WINS", "DRAWS"])
            sim_table_stats.add_row([self.num_simulations, round(
                total_simulation_time, 3), count_home_outcome['wins'], count_away_outcome['wins'], count_away_outcome['draws']])
            sim_table_stats.add_row(["-", "-", str(p_home_win) +
                                    "%", str(p_away_win)+"%", str(p_draw)+"%"])

            # print(sim_table_stats)
            # print(score_matrix)

            max_score = 6
            result_occurrences = self.build_result_occurrences(
                score_matrix=score_matrix, max_score=max_score)

            score_matrix_occurrences = result_occurrences['result_occurrences']

            print("**********************************")
            print("*                                *")
            print("*  SCORE MATRIX OCCURRENCES  *")
            print("*                                *")
            print("**********************************")
            '''
            +---+---+---+---+---+---+---+
            |   | 0 | 1 | 2 | 3 | 4 | 5 |
            +---+---+---+---+---+---+---+
            | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
            | 1 | 0 | 2 | 0 | 2 | 0 | 0 |
            | 2 | 1 | 1 | 0 | 0 | 0 | 0 |
            | 3 | 0 | 0 | 0 | 0 | 1 | 0 |
            | 4 | 2 | 0 | 1 | 0 | 0 | 0 |
            | 5 | 1 | 0 | 1 | 0 | 0 | 0 |
            +---+---+---+---+---+---+---+
            '''
            score_matrix_occ_print = PrettyTable(
                [" "] + [x for x in range(max_score)])

            for i in range(len(score_matrix_occurrences)):
                score_matrix_occ_print.add_row(
                    [i] + [score_matrix_occurrences[i][j] for j in range(len(score_matrix_occurrences))])

            print(score_matrix_occ_print)

            score_matrix_probability = self.build_score_matrix_probability(
                score_matrix_occurrences=score_matrix_occurrences, num_simulations=self.num_simulations)

            score_home_probability = score_matrix_probability['home_score_occurrence_probability']
            score_away_probability = score_matrix_probability['away_score_occurrence_probability']
            score_matrix_probability = score_matrix_probability['score_matrix_probability']

            print("**********************************")
            print("*                                *")
            print("*  SCORE MATRIX (% PROBABILITY)  *")
            print("*                                *")
            print("**********************************")
            '''
            #Home
            |
            v
            +---+------+------+-----+------+-----+-----+
            |   |  0   |  1   |  2  |  3   |  4  |  5  | <- Away
            +---+------+------+-----+------+-----+-----+
            | 0 | 0.0  | 0.0  | 6.7 | 13.3 | 0.0 | 0.0 |
            | 1 | 6.7  | 13.3 | 0.0 | 6.7  | 0.0 | 6.7 |
            | 2 | 0.0  | 13.3 | 0.0 | 0.0  | 0.0 | 0.0 |
            | 3 | 0.0  | 6.7  | 0.0 | 0.0  | 0.0 | 0.0 |
            | 4 | 13.3 | 0.0  | 0.0 | 0.0  | 0.0 | 0.0 |
            | 5 | 0.0  | 6.7  | 0.0 | 0.0  | 0.0 | 0.0 |
            +---+------+------+-----+------+-----+-----+
            '''

            score_matrix_print = PrettyTable(
                [" "] + [x for x in range(max_score)])

            for i in range(len(score_matrix_probability)):
                score_matrix_print.add_row(
                    [i] + [round(score_matrix_probability[i][j]*100, 2) for j in range(len(score_matrix_probability))])
            print(score_matrix_print)

            game_ids.append(game_id)
            team_homes.append(team_home)
            team_aways.append(team_away)
            match_days.append(matchday)
            p_home_wins.append(p_home_win)
            p_away_wins.append(p_away_win)
            p_draws.append(p_draw)
            p_scoring_home.append(score_home_probability)
            p_scoring_away.append(score_away_probability)

        print("Len Game IDs: "+str(len(game_ids)))
        return pd.DataFrame(
            {
                'game_id': game_ids,
                'matchday': match_days,
                'team_homes': team_homes,
                'team_aways': team_aways,
                'p_home_win': p_home_wins,
                'p_away_win': p_away_wins,
                'p_draw': p_draws,
                'p_home_score': p_scoring_home,
                'p_home_away': p_scoring_away
            }
        ).round(3)

    def start_simulation_new(self):
        for i in range(0, len(self.df_matches)):
            print("* Game #", i+1, "*")
            print("* Home team:", self.df_matches.iloc[i]['team_home'])
            print("* Away team:", self.df_matches.iloc[i]['team_away'])
            print("* Home team Indicator:",
                  self.df_matches.iloc[i]['indicator_home'])
            print("* Away team Indicator:", self.df_matches.iloc[i]['xG_away'])
            input_game_id = self.df_matches.iloc[i]['game_id']
            input_home_team = self.df_matches.iloc[i]['team_home']
            # input_home_team_xg = self.df_matches.iloc[i]['xG_home']
            input_home_team_xg = self.df_matches.iloc[i]['indicator_home']
            input_away_team = self.df_matches.iloc[i]['team_away']
            # input_away_team_xg = self.df_matches.iloc[i]['xG_away']
            input_away_team_xg = self.df_matches.iloc[i]['indicator_away']
            goal_home = self.df_matches.iloc[i]['Gol_x']
            goal_away = self.df_matches.iloc[i]['Gol_y']
            # print the simulation table and run simulations

            home_info = {
                'team': input_home_team,
                'indicator': input_home_team_xg
            }

            away_info = {
                'team': input_away_team,
                'indicator': input_home_team_xg
            }
            print("********************")
            print("*                  *")
            print("* SIMULATION TABLE *")
            print("*                  *")
            print("********************")
            count_home_wins = 0
            count_home_loss = 0
            count_away_wins = 0
            count_away_loss = 0
            count_draws = 0
            score_mat = []
            tot_sim_time = 0
            sim_table = PrettyTable(["SIMULATION #", "SIMULATION TIME (s)", input_home_team,
                                    input_away_team, "HOME WIN", "AWAY WIN", "DRAW", "SCORE MARGIN"])

            for i in range(self.num_simulations):
                # get simulation start time
                start_time = time.time()
                # run the sim - generate a random Poisson distribution
                target_home_goals_scored = np.random.poisson(
                    input_home_team_xg)
                target_away_goals_scored = np.random.poisson(
                    input_away_team_xg)
                home_win = 0
                away_win = 0
                draw = 0
                margin = 0
                # if more goals for home team => home team wins
                if target_home_goals_scored > target_away_goals_scored:
                    count_home_wins += 1
                    count_away_loss += 1
                    home_win = 1
                    margin = target_home_goals_scored - target_away_goals_scored
                # if more goals for away team => away team wins
                elif target_home_goals_scored < target_away_goals_scored:
                    count_away_wins += 1
                    count_home_loss += 1
                    away_win = 1
                    margin = target_away_goals_scored - target_home_goals_scored
                elif target_home_goals_scored == target_away_goals_scored:
                    draw = 1
                    count_draws += 1
                    margin = target_away_goals_scored - target_home_goals_scored
                # add score to score matrix
                score_mat.append(
                    (target_home_goals_scored, target_away_goals_scored))
                # get end time
                end_time = time.time()
                # add the time to the total simulation time
                tot_sim_time += round((end_time - start_time), 5)
                # add the info to the simulation table
                sim_table.add_row([i+1, round((end_time - start_time), 5), target_home_goals_scored,
                                   target_away_goals_scored, home_win, away_win, draw, margin])
            print(sim_table)

            self.p_home_win = round(
                (count_home_wins/self.num_simulations * 100), 2)
            self.p_away_win = round(
                (count_away_wins/self.num_simulations * 100), 2)
            self.p_draw = round(
                (count_draws/self.num_simulations * 100), 2)

            # print the simulation statistics
            print("*************")
            print("*           *")
            print("* SIM STATS *")
            print("*           *")
            print("*************")
            sim_table_stats = PrettyTable(
                ["Total # of sims", "Total time (s) for sims", "HOME WINS", "AWAY WINS", "DRAWS"])
            sim_table_stats.add_row([self.num_simulations, round(
                tot_sim_time, 3), count_home_wins, count_away_wins, count_draws])
            sim_table_stats.add_row(["-", "-", str(self.p_home_win) +
                                    "%", str(self.p_away_win)+"%", str(self.p_draw)+"%"])
            # print(sim_table_stats)
            print(score_mat)

            # get the score matrix
            total_scores = len(score_mat)
            max_score = 5
            assemble_scores = [
                [0 for x in range(max_score)] for y in range(max_score)]
            for i in range(total_scores):
                if score_mat[i][0] == 0 and score_mat[i][1] == 0:
                    assemble_scores[0][0] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 1:
                    assemble_scores[0][1] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 2:
                    assemble_scores[0][2] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 3:
                    assemble_scores[0][3] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 4:
                    assemble_scores[0][4] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 0:
                    assemble_scores[1][0] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 1:
                    assemble_scores[1][1] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 2:
                    assemble_scores[1][2] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 3:
                    assemble_scores[1][3] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 4:
                    assemble_scores[1][4] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 0:
                    assemble_scores[2][0] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 1:
                    assemble_scores[2][1] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 2:
                    assemble_scores[2][2] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 3:
                    assemble_scores[2][3] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 4:
                    assemble_scores[2][4] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 0:
                    assemble_scores[3][0] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 1:
                    assemble_scores[3][1] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 2:
                    assemble_scores[3][2] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 3:
                    assemble_scores[3][3] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 4:
                    assemble_scores[3][4] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 0:
                    assemble_scores[4][0] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 1:
                    assemble_scores[4][1] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 2:
                    assemble_scores[4][2] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 3:
                    assemble_scores[4][3] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 4:
                    assemble_scores[4][4] += 1

            # calculate percentages and print the score matrix
            print("**********************************")
            print("*                                *")
            print("*  SCORE MATRIX (% PROBABILITY)  *")
            print("*                                *")
            print("**********************************")
            score_matrix = PrettyTable([" ", 0, 1, 2, 3, 4])
            score_matrix.add_row([0, round(assemble_scores[0][0]/self.num_simulations*100, 2), round(assemble_scores[0][1]/self.num_simulations*100, 2), round(
                assemble_scores[0][2]/self.num_simulations*100, 2), round(assemble_scores[0][3]/self.num_simulations*100, 2), round(assemble_scores[0][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([1, round(assemble_scores[1][0]/self.num_simulations*100, 2), round(assemble_scores[1][1]/self.num_simulations*100, 2), round(
                assemble_scores[1][2]/self.num_simulations*100, 2), round(assemble_scores[1][3]/self.num_simulations*100, 2), round(assemble_scores[1][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([2, round(assemble_scores[2][0]/self.num_simulations*100, 2), round(assemble_scores[2][1]/self.num_simulations*100, 2), round(
                assemble_scores[2][2]/self.num_simulations*100, 2), round(assemble_scores[2][3]/self.num_simulations*100, 2), round(assemble_scores[2][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([3, round(assemble_scores[3][0]/self.num_simulations*100, 2), round(assemble_scores[3][1]/self.num_simulations*100, 2), round(
                assemble_scores[3][2]/self.num_simulations*100, 2), round(assemble_scores[3][3]/self.num_simulations*100, 2), round(assemble_scores[3][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([4, round(assemble_scores[4][0]/self.num_simulations*100, 2), round(assemble_scores[4][1]/self.num_simulations*100, 2), round(
                assemble_scores[4][2]/self.num_simulations*100, 2), round(assemble_scores[4][3]/self.num_simulations*100, 2), round(assemble_scores[4][4]/self.num_simulations*100, 2)])
            print(score_matrix)

            # calculate expected Pts and print a summary
            self.home_xPts = (self.p_home_win / 100) * 3.0 + \
                (self.p_draw / 100) * 1.0 + (self.p_away_win / 100) * 0.0
            self.away_xPts = (self.p_away_win / 100) * 3.0 + \
                (self.p_draw / 100) * 1.0 + (self.p_away_win / 100) * 0.0
            print("**********************************")
            print("*                                *")
            print("*             SUMMARY            *")
            print("*                                *")
            print("**********************************")
            print(input_home_team, "win probability %:",
                  self.p_home_win, "xPts =", round(self.home_xPts, 2))
            print(input_away_team, "win probability %:",
                  self.p_away_win, "xPts =", round(self.away_xPts, 2))
            print("Draw probability %:", self.p_draw)
            print("**********************************")
            result_tuple_csv = (input_home_team, round(self.p_home_win, 2), round(
                self.home_xPts, 2), goal_home, input_away_team, round(self.p_away_win, 2), round(self.away_xPts, 2), goal_away,
                "\n")
            metrics_filename = '_'.join(self.considered_metrics)
            filename = f'MonteCarloMatchSimResults_{self.num_simulations}_{metrics_filename}.csv'
            with open(filename, "a+") as myfile:
                myfile.write(",".join(map(str, result_tuple_csv)))
            ret_dict['game_id'].append(input_game_id)
            ret_dict['p_win_home'].append(self.p_home_win)
            ret_dict['p_win_away'].append(self.p_away_win)
            ret_dict['p_draw'].append(self.p_draw)
            ret_dict['xPts_home'].append(self.home_xPts)
            ret_dict['xPts_away'].append(self.away_xPts)
            ret_dict['goal_home'].append(goal_home)
            ret_dict['goal_away'].append(goal_away)

        return pd.DataFrame(ret_dict)

    def start_simulation(self):
        ret_dict = {
            'game_id': [],
            'p_win_home': [],
            'p_win_away': [],
            'p_draw': [],
            'xPts_home': [],
            'xPts_away': [],
            'goal_home': [],
            'goal_away': []
        }
        for i in range(0, len(self.df_matches)):
            print("* Game #", i+1, "*")
            print("* Home team:", self.df_matches.iloc[i]['team_home'])
            print("* Away team:", self.df_matches.iloc[i]['team_away'])
            print("* Home team xG:", self.df_matches.iloc[i]['xG_home'])
            print("* Away team xG:", self.df_matches.iloc[i]['xG_away'])
            input_game_id = self.df_matches.iloc[i]['game_id']
            input_home_team = self.df_matches.iloc[i]['team_home']
            # input_home_team_xg = self.df_matches.iloc[i]['xG_home']
            input_home_team_xg = self.df_matches.iloc[i]['indicator_home']
            input_away_team = self.df_matches.iloc[i]['team_away']
            # input_away_team_xg = self.df_matches.iloc[i]['xG_away']
            input_away_team_xg = self.df_matches.iloc[i]['indicator_away']
            goal_home = self.df_matches.iloc[i]['Gol_x']
            goal_away = self.df_matches.iloc[i]['Gol_y']
            # print the simulation table and run simulations
            print("********************")
            print("*                  *")
            print("* SIMULATION TABLE *")
            print("*                  *")
            print("********************")
            count_home_wins = 0
            count_home_loss = 0
            count_away_wins = 0
            count_away_loss = 0
            count_draws = 0
            score_mat = []
            tot_sim_time = 0
            sim_table = PrettyTable(["SIMULATION #", "SIMULATION TIME (s)", input_home_team,
                                    input_away_team, "HOME WIN", "AWAY WIN", "DRAW", "SCORE MARGIN"])

            for i in range(self.num_simulations):
                # get simulation start time
                start_time = time.time()
                # run the sim - generate a random Poisson distribution
                target_home_goals_scored = np.random.poisson(
                    input_home_team_xg)
                target_away_goals_scored = np.random.poisson(
                    input_away_team_xg)
                home_win = 0
                away_win = 0
                draw = 0
                margin = 0
                # if more goals for home team => home team wins
                if target_home_goals_scored > target_away_goals_scored:
                    count_home_wins += 1
                    count_away_loss += 1
                    home_win = 1
                    margin = target_home_goals_scored - target_away_goals_scored
                # if more goals for away team => away team wins
                elif target_home_goals_scored < target_away_goals_scored:
                    count_away_wins += 1
                    count_home_loss += 1
                    away_win = 1
                    margin = target_away_goals_scored - target_home_goals_scored
                elif target_home_goals_scored == target_away_goals_scored:
                    draw = 1
                    count_draws += 1
                    margin = target_away_goals_scored - target_home_goals_scored
                # add score to score matrix
                score_mat.append(
                    (target_home_goals_scored, target_away_goals_scored))
                # get end time
                end_time = time.time()
                # add the time to the total simulation time
                tot_sim_time += round((end_time - start_time), 5)
                # add the info to the simulation table
                sim_table.add_row([i+1, round((end_time - start_time), 5), target_home_goals_scored,
                                   target_away_goals_scored, home_win, away_win, draw, margin])
            print(sim_table)

            self.p_home_win = round(
                (count_home_wins/self.num_simulations * 100), 2)
            self.p_away_win = round(
                (count_away_wins/self.num_simulations * 100), 2)
            self.p_draw = round(
                (count_draws/self.num_simulations * 100), 2)

            # print the simulation statistics
            print("*************")
            print("*           *")
            print("* SIM STATS *")
            print("*           *")
            print("*************")
            sim_table_stats = PrettyTable(
                ["Total # of sims", "Total time (s) for sims", "HOME WINS", "AWAY WINS", "DRAWS"])
            sim_table_stats.add_row([self.num_simulations, round(
                tot_sim_time, 3), count_home_wins, count_away_wins, count_draws])
            sim_table_stats.add_row(["-", "-", str(self.p_home_win) +
                                    "%", str(self.p_away_win)+"%", str(self.p_draw)+"%"])
            print(sim_table_stats)
            # print(score_mat)

            # get the score matrix
            total_scores = len(score_mat)
            max_score = 5
            assemble_scores = [
                [0 for x in range(max_score)] for y in range(max_score)]
            for i in range(total_scores):
                if score_mat[i][0] == 0 and score_mat[i][1] == 0:
                    assemble_scores[0][0] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 1:
                    assemble_scores[0][1] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 2:
                    assemble_scores[0][2] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 3:
                    assemble_scores[0][3] += 1
                elif score_mat[i][0] == 0 and score_mat[i][1] == 4:
                    assemble_scores[0][4] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 0:
                    assemble_scores[1][0] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 1:
                    assemble_scores[1][1] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 2:
                    assemble_scores[1][2] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 3:
                    assemble_scores[1][3] += 1
                elif score_mat[i][0] == 1 and score_mat[i][1] == 4:
                    assemble_scores[1][4] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 0:
                    assemble_scores[2][0] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 1:
                    assemble_scores[2][1] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 2:
                    assemble_scores[2][2] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 3:
                    assemble_scores[2][3] += 1
                elif score_mat[i][0] == 2 and score_mat[i][1] == 4:
                    assemble_scores[2][4] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 0:
                    assemble_scores[3][0] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 1:
                    assemble_scores[3][1] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 2:
                    assemble_scores[3][2] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 3:
                    assemble_scores[3][3] += 1
                elif score_mat[i][0] == 3 and score_mat[i][1] == 4:
                    assemble_scores[3][4] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 0:
                    assemble_scores[4][0] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 1:
                    assemble_scores[4][1] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 2:
                    assemble_scores[4][2] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 3:
                    assemble_scores[4][3] += 1
                elif score_mat[i][0] == 4 and score_mat[i][1] == 4:
                    assemble_scores[4][4] += 1

            # calculate percentages and print the score matrix
            print("**********************************")
            print("*                                *")
            print("*  SCORE MATRIX (% PROBABILITY)  *")
            print("*                                *")
            print("**********************************")
            score_matrix = PrettyTable([" ", 0, 1, 2, 3, 4])
            score_matrix.add_row([0, round(assemble_scores[0][0]/self.num_simulations*100, 2), round(assemble_scores[0][1]/self.num_simulations*100, 2), round(
                assemble_scores[0][2]/self.num_simulations*100, 2), round(assemble_scores[0][3]/self.num_simulations*100, 2), round(assemble_scores[0][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([1, round(assemble_scores[1][0]/self.num_simulations*100, 2), round(assemble_scores[1][1]/self.num_simulations*100, 2), round(
                assemble_scores[1][2]/self.num_simulations*100, 2), round(assemble_scores[1][3]/self.num_simulations*100, 2), round(assemble_scores[1][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([2, round(assemble_scores[2][0]/self.num_simulations*100, 2), round(assemble_scores[2][1]/self.num_simulations*100, 2), round(
                assemble_scores[2][2]/self.num_simulations*100, 2), round(assemble_scores[2][3]/self.num_simulations*100, 2), round(assemble_scores[2][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([3, round(assemble_scores[3][0]/self.num_simulations*100, 2), round(assemble_scores[3][1]/self.num_simulations*100, 2), round(
                assemble_scores[3][2]/self.num_simulations*100, 2), round(assemble_scores[3][3]/self.num_simulations*100, 2), round(assemble_scores[3][4]/self.num_simulations*100, 2)])
            score_matrix.add_row([4, round(assemble_scores[4][0]/self.num_simulations*100, 2), round(assemble_scores[4][1]/self.num_simulations*100, 2), round(
                assemble_scores[4][2]/self.num_simulations*100, 2), round(assemble_scores[4][3]/self.num_simulations*100, 2), round(assemble_scores[4][4]/self.num_simulations*100, 2)])
            print(score_matrix)

            # calculate expected Pts and print a summary
            self.home_xPts = (self.p_home_win / 100) * 3.0 + \
                (self.p_draw / 100) * 1.0 + (self.p_away_win / 100) * 0.0
            self.away_xPts = (self.p_away_win / 100) * 3.0 + \
                (self.p_draw / 100) * 1.0 + (self.p_away_win / 100) * 0.0
            print("**********************************")
            print("*                                *")
            print("*             SUMMARY            *")
            print("*                                *")
            print("**********************************")
            print(input_home_team, "win probability %:",
                  self.p_home_win, "xPts =", round(self.home_xPts, 2))
            print(input_away_team, "win probability %:",
                  self.p_away_win, "xPts =", round(self.away_xPts, 2))
            print("Draw probability %:", self.p_draw)
            print("**********************************")
            result_tuple_csv = (input_home_team, round(self.p_home_win, 2), round(
                self.home_xPts, 2), goal_home, input_away_team, round(self.p_away_win, 2), round(self.away_xPts, 2), goal_away,
                "\n")
            metrics_filename = '_'.join(self.considered_metrics)
            filename = f'MonteCarloMatchSimResults_{self.num_simulations}_{metrics_filename}.csv'
            with open(filename, "a+") as myfile:
                myfile.write(",".join(map(str, result_tuple_csv)))
            ret_dict['game_id'].append(input_game_id)
            ret_dict['p_win_home'].append(self.p_home_win)
            ret_dict['p_win_away'].append(self.p_away_win)
            ret_dict['p_draw'].append(self.p_draw)
            ret_dict['xPts_home'].append(self.home_xPts)
            ret_dict['xPts_away'].append(self.away_xPts)
            ret_dict['goal_home'].append(goal_home)
            ret_dict['goal_away'].append(goal_away)

        print("END SIMULATION")
        return pd.DataFrame(ret_dict)
