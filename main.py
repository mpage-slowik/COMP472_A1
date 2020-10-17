
import m_learner
from m_learner import plot_lib


if __name__ == "__main__":
    print("Running the main.py file")
    dist_1 = m_learner.calculate_distribution(1)
    dist_2 = m_learner.calculate_distribution(2)
    # Once built can commment out to run plotting
    print("Executing GNB")
    # m_learner.gnb_predictor(1)
    # m_learner.gnb_predictor(2)
    # print("Executing Base Decision Tree")
    # m_learner.base_dt(1)
    # m_learner.base_dt(2)
    # print("Executing Best Decision Tree")
    # m_learner.best_dt(1)
    # m_learner.best_dt(2)
    # print("Executing perceptron default")
    # m_learner.default_perceptron(1)
    # m_learner.default_perceptron(2)
    # print("Executing base multi layered perceptron")
    # m_learner.base_multi_layered_perceptron(1)
    # m_learner.base_multi_layered_perceptron(2)
    print("Executing best multi layered perceptron")
    m_learner.best_multi_layered_perceptron(1)
    m_learner.best_multi_layered_perceptron(2)
    print("\nProgram done!")
    # for plotting
    plot_lib()