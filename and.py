from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd


def main(data, eta, epochs, filename, plot_name):
    """And operation using perceptron

    Args:
        data : Input Data
        eta : Learning Rate
        epochs : Number of iterations
        filename : Filename for the model
        plot_name : Plotname
    """

    df = pd.DataFrame(data)
    print(df)

    X, y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()

    save_model(model, filename)
    save_plot(df, plot_name, model)


if __name__ == "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1],
    }

    ETA = 0.3
    EPOCHS = 10

    main(
        data=AND,
        eta=ETA,
        epochs=EPOCHS,
        filename="and.model",
        plot_name="and.png",
    )
