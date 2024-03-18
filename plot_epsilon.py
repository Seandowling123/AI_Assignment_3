import matplotlib.pyplot as plt

epsilon = .5
decay_rate = .00001
final_epsilon = .0001

epsilons = []
for i in range(1000000):
    epsilons.append(epsilon)
    if epsilon > final_epsilon:
        epsilon = max(final_epsilon, epsilon * (1 / (1 + decay_rate)))
    
    
plt.figure(figsize=(12, 6))
plt.plot(epsilons, label='Epsilon', color='#2980b9', linewidth=1)
plt.title('Value of Epsilon Over Time', fontsize=14, fontfamily='serif')
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 11})
plt.xlabel('Moves Played By Q-Learning Agent During Training', fontsize=13, fontname='Times New Roman')
plt.ylabel('Value', fontsize=13, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Epsilon_Over_Time.png', bbox_inches='tight')
plt.show()