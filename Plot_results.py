import matplotlib.pyplot as plt
import pandas as pd


###############
# Tic tac toe #
###############

# Minimax vs Default
sizes = [0.234,0.572,0.194]
labels = ['Ties','Minimax wins','Default Player wins']
colors = ['#27ae60', '#e74c3c', '#2980b9']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Win Ratios of Minimax & Default Player with Minimax As Player 1', fontsize=12, fontfamily='serif')
plt.axis('equal')
plt.savefig('Plots/Minimax_P1_vs_Default.png', bbox_inches='tight')
#plt.show()
plt.close()

sizes = [0.069,0.34,0.591]
labels = ['Ties','Minimax wins','Default Player wins']
colors = ['#27ae60', '#e74c3c', '#2980b9']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Win Ratios of Minimax & Default Player with Minimax As Player 2', fontsize=12, fontfamily='serif')
plt.axis('equal')
plt.savefig('Plots/Minimax_P2_vs_Default.png', bbox_inches='tight')
#plt.show()
plt.close()


# Q-learning vs Default
df = pd.read_csv("Tictactoe_results/Q_learning_agent_P1_vs_Default Player_Results.csv")
# Plot the data
plt.figure(figsize=(10, 6))
for column in df.columns[1:]:
    plt.plot(df.index, df[column], label=column, marker='o')
    plt.ylim(0, 1)
plt.title('Win Ratios of Q-learning & Default Player with Q-learning As Player 1', fontsize=12, fontfamily='serif')
plt.axis('equal')
plt.savefig('Plots/Q_learning_agent_P1_vs_Default.png', bbox_inches='tight')
#plt.show()
plt.close()

df = pd.read_csv("Tictactoe_results/Q_learning_agent_P2_vs_Default Player_Results.csv")
for column in df.columns[1:]:
    plt.plot(df.index, df[column], label=column, marker='o')
plt.ylim(.5, 1)
plt.title('Win Ratios of Q-learning & Default Player with Q-learning As Player 2', fontsize=12, fontfamily='serif')
plt.axis('equal')
plt.savefig('Plots/Q_learning_agent_P2_vs_Default.png', bbox_inches='tight')
#plt.show()
plt.close()