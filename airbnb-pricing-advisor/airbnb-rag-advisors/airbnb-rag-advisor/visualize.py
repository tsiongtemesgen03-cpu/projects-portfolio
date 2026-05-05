import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('pricing_recommendations.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Chart 1: Price uplift by action
action_counts = df['action'].value_counts()
axes[0].bar(action_counts.index, action_counts.values, color=['#2d6a4f', '#d4a017', '#c0392b'])
axes[0].set_title('Pricing Recommendations')
axes[0].set_xlabel('Action')
axes[0].set_ylabel('Number of Listings')

# Chart 2: Baseline vs Recommended price
categories = ['Normal Weekend', 'Event Weekend']
baseline = [df['baseline_price'].mean(), df['baseline_price'].mean()]
recommended = [df[df['action']=='HOLD']['recommended_price'].mean(),
               df[df['action']=='RAISE']['recommended_price'].mean()]
x = np.arange(len(categories))
axes[1].bar(x - 0.2, baseline, 0.4, label='Baseline', color='#1a3c34')
axes[1].bar(x + 0.2, recommended, 0.4, label='Recommended', color='#a8b400')
axes[1].set_title('Baseline vs Recommended Price')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)
axes[1].legend()

# Chart 3: Distance to event vs price uplift
axes[2].scatter(df['distance_to_event_km'], df['price_uplift'], alpha=0.3, color='#1a3c34')
axes[2].set_title('Distance to Event vs Price Uplift')
axes[2].set_xlabel('Distance to Event (km)')
axes[2].set_ylabel('Price Uplift ($)')
axes[2].axhline(y=0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('pricing_charts.png', dpi=150)
print("Saved pricing_charts.png")
