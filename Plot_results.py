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
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ties'], label="Ties", marker='o')
plt.plot(df.index, df['Tictactoe_Q_learning_agent wins'], label="Q-Learning Agent Wins", marker='o')
plt.plot(df.index, df['Default Player wins'], label="Default Player Wins", marker='o')
plt.gca().set_ylim(0, 1)
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.title('Win Ratios of Q-learning & Default Player with Q-learning As Player 1', fontsize=12, fontfamily='serif')
plt.xlabel('Number of Training Iterations For Q-Learning Player', fontsize=11, fontname='Times New Roman')
plt.ylabel('Ratio', fontsize=11, fontname='Times New Roman')
plt.savefig('Plots/Q_learning_agent_P1_vs_Default.png', bbox_inches='tight')
plt.close()

df = pd.read_csv("Tictactoe_results/Q_learning_agent_P2_vs_Default Player_Results.csv")
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ties'], label="Ties", marker='o')
plt.plot(df.index, df['Tictactoe_Q_learning_agent wins'], label="Q-Learning Agent Wins", marker='o')
plt.plot(df.index, df['Default Player wins'], label="Default Player Wins", marker='o')
plt.ylim(0, 1)
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.title('Win Ratios of Q-learning & Default Player with Q-learning As Player 2', fontsize=12, fontfamily='serif')
plt.xlabel('Number of Training Iterations For Q-Learning Player', fontsize=11, fontname='Times New Roman')
plt.ylabel('Ratio', fontsize=11, fontname='Times New Roman')
plt.savefig('Plots/Q_learning_agent_P2_vs_Default.png', bbox_inches='tight')
plt.close()