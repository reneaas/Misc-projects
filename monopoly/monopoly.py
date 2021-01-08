import numpy as np
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar

class Monopoly:
    def __init__(self, number_of_players, number_of_games):
        #Basic parameters
        self.number_of_players = number_of_players
        self.player_position = np.zeros(self.number_of_players, int)
        self.player_money = np.zeros(self.number_of_players)
        self.player_rounds = np.zeros(self.number_of_players, int)
        self.total_money = 150000 #Dollars
        self.player_money[:] = self.total_money/number_of_players #Money is evenly divided among the players
        self.number_of_games = number_of_games
        self.end_of_board_bonus = 200


        #Important positions on the boardgame
        self.go_to_jail = 30 #Position of go to jail
        self.jail = 10  #Position of jail
        self.income_tax = 4
        #self.electric_company = 12
        self.end_of_board = 39
        self.position_counter = np.zeros(self.end_of_board + 1)

    def throw_dice(self):
        dice = np.random.randint(1, 7, size = self.number_of_players) #Each player throws dice
        self.player_position += dice   #Each player moves their designated step forward from their current position

        #First check if players have reached the end of the board:
        end_of_board_index = np.where(self.player_position > self.end_of_board)
        self.player_position[end_of_board_index] = self.end_of_board - self.player_position[end_of_board_index] #begin anew.
        #self.player_money[end_of_board_index] += self.end_of_board_bonus
        self.player_rounds[end_of_board_index] += 1;

        #Check if player must go to jail
        go_to_jail_index = np.where(self.player_position == self.go_to_jail)
        self.player_position[go_to_jail_index] = self.jail

        #Check if player must pay income tax
        income_tax_index = np.where(self.player_position == self.income_tax)
        self.player_money[income_tax_index] -= 0.5*self.player_money[income_tax_index] #Pay 10 percent of money in taxes.

        #add up position contribution:
        self.position_counter[self.player_position] += 1;
        #print(np.max(self.player_money))


    def play_game(self):
        bar = IncrementalBar("Progress", max = self.number_of_games)
        for i in range(self.number_of_games):
            self.player_money[:] = self.total_money/self.number_of_players #Money is evenly divided among the players
            np.random.seed(i) #Change seed per to improve statistics
            bar.next()
            while ((self.player_money > 1).all() == 1):
                self.throw_dice()
        bar.finish()


    def compute_probabilities(self):
        self.probabilities = self.position_counter/sum(self.position_counter)
        #print(sum(self.probabilities))

    def plot_probabilities(self):
        positions = np.linspace(0, self.end_of_board+1, self.end_of_board+1)

        plt.plot(positions, self.probabilities)
        #plt.hist(self.position_counter, density = True, bins = positions)
        plt.xlabel("Position")
        plt.ylabel("Probability")
        plt.title("After " + str(np.max(self.number_of_games)) + " games")
        plt.savefig("Probabilities.pdf", dpi = 1000)
        plt.show()

