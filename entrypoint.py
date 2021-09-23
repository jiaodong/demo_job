import argparse
from ray import tune
import time


def main(num_steps):

    def objective(step, alpha, beta):
        return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


    def training_function(config):
        # Hyperparameters
        alpha, beta = config["alpha"], config["beta"]
        for step in range(num_steps):
	    # Add sleep here to make tune run longer if needed
            time.sleep(1)
            # Iterative training function - can be any arbitrary training procedure.
            intermediate_score = objective(step, alpha, beta)
            # Feed the score back back to Tune.
            tune.report(mean_loss=intermediate_score)


    analysis = tune.run(
        training_function,
        config={
            "alpha": tune.grid_search([0.001, 0.01, 0.1]),
            "beta": tune.choice([1, 2, 3])
        })

    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-steps",
        required=False,
        type=int,
        default=10,
        help="num steps")
    
    args, _ = parser.parse_known_args()
    main(args.num_steps)
    

