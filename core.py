from services import TradingPipeline
from models import DecisionTreeClassModel


def main():
    pipeline = TradingPipeline(
        data_dir="data",
        save_dir="predictions",
        eval_dir="evaluations"
    )
    
    print("Starting S&P 500 Trading Pipeline...")
    

    strategies = [
        DecisionTreeClassModel(max_depth=5, criterion="gini"),
    ]

    for strategy in strategies:
        try:
            pipeline.run(strategy)
        except Exception as e:
            print(f"Error running strategy {strategy.name}: {e}")

    print("Pipeline execution finished.")


if __name__ == "__main__":
    main()
