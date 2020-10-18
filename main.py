
import m_learner
from m_learner import plot_lib


if __name__ == "__main__":
    print("Running the main.py file")
    # dist_1 = m_learner.calculate_distribution(1)
    # dist_2 = m_learner.calculate_distribution(2)
    # Once built can commment out to run plotting
    print("Executing GNB")
    m_learner.gnb_predictor(1,"./data/train_1.csv","./output/GNB-DS1.csv","./data/test_with_label_1.csv")
    m_learner.gnb_predictor(2,"./data/train_2.csv","./output/GNB-DS2.csv","./data/test_with_label_2.csv")
    print("Executing Base Decision Tree")
    m_learner.base_dt(1,"./data/train_1.csv","./output/Base-DT-DS1.csv","./data/test_with_label_1.csv")
    m_learner.base_dt(2,"./data/train_2.csv","./output/Base-DT-DS2.csv","./data/test_with_label_2.csv")
    print("Executing Best Decision Tree")
    m_learner.best_dt(1,"./data/train_1.csv","./output/Best-DT-DS1.csv","./data/test_with_label_1.csv")
    m_learner.best_dt(2,"./data/train_2.csv","./output/Best-DT-DS2.csv","./data/test_with_label_2.csv")
    print("Executing perceptron default")
    m_learner.default_perceptron(1,"./data/train_1.csv","./output/PER-DS1.csv","./data/test_with_label_1.csv")
    m_learner.default_perceptron(2,"./data/train_2.csv","./output/PER-DS2.csv","./data/test_with_label_2.csv")
    print("Executing base multi layered perceptron")
    m_learner.base_multi_layered_perceptron(1,"./data/train_1.csv","./output/Base-MLP-DS1.csv","./data/test_with_label_1.csv")
    m_learner.base_multi_layered_perceptron(2,"./data/train_2.csv","./output/Base-MLP-DS2.csv","./data/test_with_label_2.csv")
    print("Executing best multi layered perceptron")
    m_learner.best_multi_layered_perceptron(1,"./data/train_1.csv","./output/Best-MLP-DS1.csv","./data/test_with_label_1.csv")
    m_learner.best_multi_layered_perceptron(2,"./data/train_2.csv","./output/Best-MLP-DS2.csv","./data/test_with_label_2.csv")
    print("\nProgram done!")
    # for plotting
    # plot_lib()