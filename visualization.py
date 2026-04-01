import matplotlib.pyplot as plt

def plot_prediction_probabilities(prediction_prob):
    labels = ['Fake News', 'True News']
    plt.bar(labels, prediction_prob, color=['red', 'green'])
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.ylim(0, 1)
    plt.show()
