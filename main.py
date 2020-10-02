
import m_learner


if __name__ == "__main__":
    print("Running the main.py file")
    # dist_1 = m_learner.calculate_distribution(1)
    # dist_2 = m_learner.calculate_distribution(2)
    # m_learner.gnb_predictor(1)
    # m_learner.gnb_predictor(2)
    # m_learner.base_dt(1)
    # m_learner.base_dt(2)
    # m_learner.best_dt(1)
    # m_learner.best_dt(2)
    print("Executing perceptron default")
    m_learner.default_perceptron(1)
    m_learner.default_perceptron(2)
    print("Executing base multi layered perceptron")
    m_learner.base_multi_layered_perceptron(1)
    m_learner.base_multi_layered_perceptron(2)
    print("Program done!")