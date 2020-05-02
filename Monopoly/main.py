from monopoly import Monopoly

monopoly_simulator = Monopoly(number_of_players = 4, number_of_games = 100000)

monopoly_simulator.play_game()
monopoly_simulator.compute_probabilities()
monopoly_simulator.plot_probabilities()
